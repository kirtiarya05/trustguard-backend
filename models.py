from sqlalchemy import Column, Integer, String
from database import Base

class QueryLog(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    score = Column(Integer)
    category = Column(String)