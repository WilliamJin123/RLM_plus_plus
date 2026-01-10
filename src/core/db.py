from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from src.config.config import config

Base = declarative_base()

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    start_index = Column(Integer)
    end_index = Column(Integer)
    
    # Optional: link to a summary if needed directly, 
    # though usually summaries cover multiple chunks.
    # summary_id = Column(Integer, ForeignKey('summaries.id'))

class Summary(Base):
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True)
    summary_text = Column(Text, nullable=False)
    level = Column(Integer, default=0) # 0=leaf (summarizes chunks), 1=branch, etc.
    parent_id = Column(Integer, ForeignKey('summaries.id'), nullable=True)
    
    # For Level 0 summaries, we might want to know which chunks they cover
    # This could be a many-to-many or we can store range of chunk IDs
    chunk_ids = Column(Text) # Storing as comma-separated IDs or JSON for simplicity in SQLite

    def get_chunk_id_list(self):
        if not self.chunk_ids:
            return []
        return [int(id_.strip()) for id_ in self.chunk_ids.split(",") if id_.strip()]

    parent = relationship("Summary", remote_side=[id], backref="children")

engine = create_engine(f"sqlite:///{config.DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(db_path: str = None):
    global engine
    if db_path:
        # Create new engine and rebind session factory
        engine = create_engine(f"sqlite:///{db_path}")
        SessionLocal.configure(bind=engine)
        print(f"Database initialized at {db_path}")
    else:
        # Just use the default engine
        print(f"Database initialized at {config.DB_PATH}")

    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {config.DB_PATH}")
