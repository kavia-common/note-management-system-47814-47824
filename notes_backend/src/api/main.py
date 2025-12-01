from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, create_engine, select, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

import os
import secrets

# Settings
class Settings(BaseModel):
    app_name: str = "Notes API"
    version: str = "0.1.0"
    description: str = "A FastAPI backend for a note management system with JWT-based authentication."
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./notes.db")
    jwt_secret: str = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
    jwt_alg: str = os.getenv("JWT_ALG", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
    cors_allow_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

settings = Settings()

# Database setup
Base = declarative_base()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

# Password hashing & JWT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_alg)
    return encoded_jwt

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])

# SQLAlchemy models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    notes = relationship("Note", back_populates="owner", cascade="all, delete-orphan")

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    owner = relationship("User", back_populates="notes")

# Pydantic schemas
class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")

class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")

class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        from_attributes = True

class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Note title")
    content: Optional[str] = Field("", description="Note content")

class NoteCreate(NoteBase):
    pass

class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Note title")
    content: Optional[str] = Field(None, description="Note content")

class NoteOut(BaseModel):
    id: int
    title: str
    content: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# FastAPI app with metadata and tags
openapi_tags = [
    {"name": "Health", "description": "Service health and metadata."},
    {"name": "Auth", "description": "User registration and login."},
    {"name": "Notes", "description": "Operations on notes (CRUD). Requires authentication."},
]

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    openapi_tags=openapi_tags,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_allow_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get current user from token
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = int(sub)
    except (JWTError, ValueError):
        raise credentials_exception
    user = db.get(User, user_id)
    if user is None:
        raise credentials_exception
    return user

# Routes

# PUBLIC_INTERFACE
@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"message": "Healthy", "service": settings.app_name, "version": settings.version}

# PUBLIC_INTERFACE
@app.post("/auth/register", tags=["Auth"], response_model=UserOut, status_code=status.HTTP_201_CREATED, summary="Register a user")
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user with email and password.

    Body:
      - email: valid email
      - password: min length 6

    Returns: UserOut
    """
    existing = db.execute(select(User).where(User.email == user_in.email.lower())).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    user = User(email=user_in.email.lower(), password_hash=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# PUBLIC_INTERFACE
@app.post("/auth/login", tags=["Auth"], response_model=Token, summary="Login and receive JWT")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login a user using OAuth2PasswordRequestForm.

    Form fields:
      - username: email
      - password: password

    Returns:
      - access_token: JWT
      - token_type: "bearer"
    """
    user = db.execute(select(User).where(User.email == form_data.username.lower())).scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer")

# PUBLIC_INTERFACE
@app.get("/notes", tags=["Notes"], response_model=List[NoteOut], summary="List notes")
def list_notes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all notes for the authenticated user."""
    notes = db.execute(select(Note).where(Note.owner_id == current_user.id).order_by(Note.updated_at.desc())).scalars().all()
    return notes

# PUBLIC_INTERFACE
@app.post("/notes", tags=["Notes"], response_model=NoteOut, status_code=status.HTTP_201_CREATED, summary="Create note")
def create_note(note_in: NoteCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new note for the authenticated user."""
    note = Note(title=note_in.title, content=note_in.content or "", owner_id=current_user.id)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note

# PUBLIC_INTERFACE
@app.get("/notes/{note_id}", tags=["Notes"], response_model=NoteOut, summary="Get note by ID")
def get_note(note_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve a single note owned by the authenticated user."""
    note = db.get(Note, note_id)
    if not note or note.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return note

# PUBLIC_INTERFACE
@app.put("/notes/{note_id}", tags=["Notes"], response_model=NoteOut, summary="Update note by ID")
def update_note(note_id: int, note_in: NoteUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a note's title/content for the authenticated user."""
    note = db.get(Note, note_id)
    if not note or note.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    if note_in.title is not None:
        if len(note_in.title.strip()) == 0:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Title cannot be empty")
        note.title = note_in.title
    if note_in.content is not None:
        note.content = note_in.content
    db.add(note)
    db.commit()
    db.refresh(note)
    return note

# PUBLIC_INTERFACE
@app.delete("/notes/{note_id}", tags=["Notes"], status_code=status.HTTP_204_NO_CONTENT, summary="Delete note by ID")
def delete_note(note_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a note owned by the authenticated user."""
    note = db.get(Note, note_id)
    if not note or note.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    db.delete(note)
    db.commit()
    return None
