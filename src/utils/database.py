"""
Database Utilities

This module provides database management using SQLAlchemy for storing matches,
odds, predictions, and team data.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import StaticPool
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)

Base = declarative_base()


# Database Models

class Team(Base):
    """Team model for storing team information."""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    league = Column(String(100))
    season = Column(String(20))
    
    # Season statistics
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_scored = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    
    # Advanced stats
    avg_possession = Column(Float)
    avg_shots = Column(Float)
    avg_shots_on_target = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Team(name='{self.name}', league='{self.league}')>"


class Match(Base):
    """Match model for storing match information and results."""
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.id'))
    away_team_id = Column(Integer, ForeignKey('teams.id'))
    
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    
    competition = Column(String(100))
    season = Column(String(20))
    match_date = Column(DateTime)
    
    # Match result
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    
    # Expected goals
    home_xg = Column(Float)
    away_xg = Column(Float)
    
    # Match status
    status = Column(String(20))  # scheduled, live, finished
    
    # Additional data
    venue = Column(String(100))
    attendance = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    odds = relationship("Odds", back_populates="match", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="match", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Match(home='{self.home_team}', away='{self.away_team}', date='{self.match_date}')>"


class Odds(Base):
    """Odds model for storing bookmaker odds."""
    __tablename__ = 'odds'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    
    bookmaker = Column(String(50), nullable=False)
    
    # 1X2 odds
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    
    # Over/Under 2.5
    over_2_5_odds = Column(Float)
    under_2_5_odds = Column(Float)
    
    # Both Teams To Score
    btts_yes_odds = Column(Float)
    btts_no_odds = Column(Float)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="odds")
    
    def __repr__(self):
        return f"<Odds(bookmaker='{self.bookmaker}', match_id={self.match_id})>"


class Prediction(Base):
    """Prediction model for storing model predictions."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    
    model_type = Column(String(50), nullable=False)  # poisson, ml, ensemble
    
    # Predicted probabilities
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    
    # Expected goals
    home_expected_goals = Column(Float)
    away_expected_goals = Column(Float)
    
    # Over/Under probabilities
    over_2_5_prob = Column(Float)
    under_2_5_prob = Column(Float)
    
    # BTTS probabilities
    btts_yes_prob = Column(Float)
    btts_no_prob = Column(Float)
    
    # Value bets identified
    has_value_bets = Column(Boolean, default=False)
    value_bets_json = Column(Text)  # JSON string of value bets
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)
    
    # Relationships
    match = relationship("Match", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(model='{self.model_type}', match_id={self.match_id})>"


class Database:
    """
    Database manager class for handling connections and operations.
    """
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """
        Initialize database connection.
        
        Args:
            database_url: Database connection URL. If None, uses SQLite in-memory
            echo: Whether to echo SQL statements
        """
        if database_url is None:
            # Use SQLite in-memory database for testing
            database_url = "sqlite:///:memory:"
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo
            )
        else:
            self.engine = create_engine(database_url, echo=echo)
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database initialized: {database_url}")
    
    def create_tables(self):
        """Create all tables in the database."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables in the database."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise
    
    def get_session(self):
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()
    
    # Team CRUD operations
    
    def add_team(self, session, team_data: Dict) -> Team:
        """
        Add a new team to the database.
        
        Args:
            session: Database session
            team_data: Dictionary with team data
        
        Returns:
            Created Team object
        """
        try:
            team = Team(**team_data)
            session.add(team)
            session.commit()
            session.refresh(team)
            logger.info(f"Team added: {team.name}")
            return team
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding team: {e}")
            raise
    
    def get_team_by_name(self, session, name: str) -> Optional[Team]:
        """
        Get a team by name.
        
        Args:
            session: Database session
            name: Team name
        
        Returns:
            Team object or None
        """
        return session.query(Team).filter(Team.name == name).first()
    
    def update_team(self, session, team_id: int, team_data: Dict):
        """
        Update team information.
        
        Args:
            session: Database session
            team_id: Team ID
            team_data: Dictionary with updated data
        """
        try:
            team = session.query(Team).filter(Team.id == team_id).first()
            if team:
                for key, value in team_data.items():
                    setattr(team, key, value)
                team.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Team updated: {team.name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating team: {e}")
            raise
    
    # Match CRUD operations
    
    def add_match(self, session, match_data: Dict) -> Match:
        """
        Add a new match to the database.
        
        Args:
            session: Database session
            match_data: Dictionary with match data
        
        Returns:
            Created Match object
        """
        try:
            match = Match(**match_data)
            session.add(match)
            session.commit()
            session.refresh(match)
            logger.info(f"Match added: {match.home_team} vs {match.away_team}")
            return match
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding match: {e}")
            raise
    
    def get_match(self, session, match_id: int) -> Optional[Match]:
        """
        Get a match by ID.
        
        Args:
            session: Database session
            match_id: Match ID
        
        Returns:
            Match object or None
        """
        return session.query(Match).filter(Match.id == match_id).first()
    
    def get_upcoming_matches(self, session, limit: int = 10) -> List[Match]:
        """
        Get upcoming matches.
        
        Args:
            session: Database session
            limit: Maximum number of matches to return
        
        Returns:
            List of Match objects
        """
        return session.query(Match).filter(
            Match.status == 'scheduled',
            Match.match_date >= datetime.utcnow()
        ).order_by(Match.match_date).limit(limit).all()
    
    # Odds CRUD operations
    
    def add_odds(self, session, odds_data: Dict) -> Odds:
        """
        Add odds to the database.
        
        Args:
            session: Database session
            odds_data: Dictionary with odds data
        
        Returns:
            Created Odds object
        """
        try:
            odds = Odds(**odds_data)
            session.add(odds)
            session.commit()
            session.refresh(odds)
            logger.info(f"Odds added for match {odds.match_id}")
            return odds
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding odds: {e}")
            raise
    
    def get_latest_odds(self, session, match_id: int, bookmaker: str = None) -> Optional[Odds]:
        """
        Get latest odds for a match.
        
        Args:
            session: Database session
            match_id: Match ID
            bookmaker: Optional bookmaker filter
        
        Returns:
            Odds object or None
        """
        query = session.query(Odds).filter(Odds.match_id == match_id)
        if bookmaker:
            query = query.filter(Odds.bookmaker == bookmaker)
        return query.order_by(Odds.timestamp.desc()).first()
    
    # Prediction CRUD operations
    
    def add_prediction(self, session, prediction_data: Dict) -> Prediction:
        """
        Add a prediction to the database.
        
        Args:
            session: Database session
            prediction_data: Dictionary with prediction data
        
        Returns:
            Created Prediction object
        """
        try:
            prediction = Prediction(**prediction_data)
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            logger.info(f"Prediction added for match {prediction.match_id}")
            return prediction
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding prediction: {e}")
            raise
    
    def get_predictions_for_match(self, session, match_id: int) -> List[Prediction]:
        """
        Get all predictions for a match.
        
        Args:
            session: Database session
            match_id: Match ID
        
        Returns:
            List of Prediction objects
        """
        return session.query(Prediction).filter(
            Prediction.match_id == match_id
        ).order_by(Prediction.timestamp.desc()).all()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create database
    db = Database()
    db.create_tables()
    
    # Get a session
    session = db.get_session()
    
    try:
        # Add teams
        team1 = db.add_team(session, {
            'name': 'Team A',
            'league': 'Premier League',
            'season': '2023/24'
        })
        
        team2 = db.add_team(session, {
            'name': 'Team B',
            'league': 'Premier League',
            'season': '2023/24'
        })
        
        # Add match
        match = db.add_match(session, {
            'home_team': 'Team A',
            'away_team': 'Team B',
            'competition': 'Premier League',
            'match_date': datetime(2024, 1, 15, 15, 0),
            'status': 'scheduled'
        })
        
        # Add odds
        odds = db.add_odds(session, {
            'match_id': match.id,
            'bookmaker': '1xBet',
            'home_odds': 2.5,
            'draw_odds': 3.2,
            'away_odds': 2.8
        })
        
        # Add prediction
        prediction = db.add_prediction(session, {
            'match_id': match.id,
            'model_type': 'poisson',
            'home_win_prob': 0.45,
            'draw_prob': 0.25,
            'away_win_prob': 0.30
        })
        
        print(f"Database operations successful!")
        print(f"Match: {match}")
        print(f"Odds: {odds}")
        print(f"Prediction: {prediction}")
    
    finally:
        session.close()
