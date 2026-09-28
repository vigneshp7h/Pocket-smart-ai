import os
import re
import json
import uuid
import shutil
import asyncio
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("No Google API key found in environment variables. Please set GOOGLE_API_KEY in your .env file.")

from google import genai

client = genai.Client(api_key=API_KEY)

class GeminiWrapper:
    def generate_content(self, contents):
        return client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents
        )

model = GeminiWrapper()

# FastAPI app initialization
app = FastAPI(title="PocketSmart: AI Budget Planner")

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories & Templating
os.makedirs("static/uploads", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- IN-MEMORY STATE STORES ----------------- #
users_db: Dict[str, dict] = {}
active_sessions: Dict[str, Any] = {}
blacklisted_tokens: set = set()
user_recommendations: Dict[str, List[Any]] = {}

# ----------------- DATA MODELS ----------------- #
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserSession:
    def __init__(self, username: str, token: str, user_data: Optional[dict] = None):
        self.username = username
        self.token = token
        self.login_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.user_data = user_data or {}

class RecommendationRecord:
    def __init__(self, username: str, rec_type: str, input_summary: str, result_summary: str, full_result: dict):
        self.id = str(uuid.uuid4())
        self.username = username
        self.recommendation_type = rec_type
        self.input_summary = input_summary
        self.result_summary = result_summary
        self.full_result = full_result
        self.timestamp = datetime.utcnow().strftime("%B %d, %Y")

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = ""

class PartyBudgetInput(BaseModel):
    total_budget: float
    party_type: str
    num_guests: int
    venue_type: Optional[str] = "Not specified"
    needs_catering: bool = False
    needs_decoration: bool = False
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = ""

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = "Not specified"

# ----------------- SECURITY & AUTH HELPERS ----------------- #
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(db: Dict[str, dict], username: str, password: str):
    user = db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return UserInDB(**user)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return None

async def get_current_user(request: Request, token: Optional[str] = None) -> UserInDB:
    if not token:
        token = await get_token(request)
    if not token or token in blacklisted_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if username in active_sessions:
            active_sessions[username].last_activity = datetime.utcnow()
        return UserInDB(**users_db[username])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate token")

async def get_current_active_user(user: UserInDB = Depends(get_current_user)) -> UserInDB:
    return user

def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    clean_text = match.group(1).strip() if match else text
    try:
        return json.loads(clean_text)
    except Exception:
        first = clean_text.find("{")
        last = clean_text.rfind("}")
        if first != -1 and last != -1:
            return json.loads(clean_text[first : last + 1])
        raise ValueError(f"Failed to parse LLM JSON: {text[:200]}")

def save_upload_file(upload_file: UploadFile) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join("static/uploads", f"{timestamp}_{upload_file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    if username not in user_recommendations:
        user_recommendations[username] = []
    
    budget = input_data.get("total_budget", 0.0)
    if recommendation_type == "home":
        in_sum = f"Budget: ₹{budget:,.0f} for Interior Styling"
        out_sum = f"Items Allocated: {len(result.get('budget_breakdown', []))} categories"
    elif recommendation_type == "party":
        in_sum = f"Budget: ₹{budget:,.0f} for {input_data.get('num_guests', 0)} guests"
        out_sum = f"Party Type: {input_data.get('party_type', 'General')}"
    else:
        in_sum = f"Budget: ₹{budget:,.0f} for {input_data.get('occasion', 'General')}"
        out_sum = f"Accessories: {len(result.get('jewelry_recommendations', []))} items"

    record = RecommendationRecord(
        username=username,
        rec_type=recommendation_type,
        input_summary=in_sum,
        result_summary=out_sum,
        full_result=result
    )
    user_recommendations[username].append(record)

# ----------------- AI RECOMMENDATION ENGINES ----------------- #
def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    try:
        prompt = f"""
I need interior design product recommendations for a home in India with a total budget of ₹{budget_input.total_budget:.2f}.
Requirements:
- {budget_input.num_lights} lights/lighting fixtures
- {budget_input.num_fans} ceiling fans
- {budget_input.num_furniture} furniture pieces
- {budget_input.num_dining_tables} dining tables

Additional rooms to consider:
{("- Living room" if budget_input.has_living_room else "")}
{("- Kitchen" if budget_input.has_kitchen else "")}
{("- Bedroom" if budget_input.has_bedroom else "")}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with product recommendations **available in India**.
Use **Indian brands and pricing**. Include **search terms** suitable for Indian shopping platforms.

Format your response strictly as JSON with this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "Lighting",
      "allocation": 1500.0,
      "items": [
        {{
          "name": "LED Bulb",
          "description": "Energy-efficient LED bulbs",
          "estimated_price": 500.0,
          "quantity": 3,
          "search_terms": "philips led bulb 9w"
        }}
      ]
    }}
  ],
  "calculation_table": [
    {{
      "category": "Lighting",
      "items_count": 3,
      "total_cost": 1500.0,
      "percentage_of_budget": 30.0
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": [
    "Prioritize essential items",
    "Look for festive discounts online"
  ]
}}
Ensure total costs stay within budget. Include realistic search terms.
"""
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        for category in result.get("budget_breakdown", []):
            for item in category.get("items", []):
                st = item.get("search_terms", "")
                if st:
                    q = urllib.parse.quote_plus(st)
                    item["shopping_links"] = {
                        "amazon": f"[https://www.amazon.in/s?k=](https://www.amazon.in/s?k=){q}",
                        "flipkart": f"[https://www.flipkart.com/search?q=](https://www.flipkart.com/search?q=){q}",
                        "ikea": f"[https://www.ikea.com/in/en/search/?q=](https://www.ikea.com/in/en/search/?q=){q}",
                        "myntra": f"[https://www.myntra.com/search?q=](https://www.myntra.com/search?q=){q}",
                        "ajio": f"[https://www.ajio.com/search/?text=](https://www.ajio.com/search/?text=){q}"
                    }
        return result
    except Exception as e:
        raise HTTPException(500, f"Error generating recommendations: {str(e)}")

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    try:
        prompt = f"""
I need party planning recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.

Party details:
- Type: {budget_input.party_type}
- Number of guests: {budget_input.num_guests}
- Venue type: {budget_input.venue_type or "Not specified"}
- Catering needed: {"Yes" if budget_input.needs_catering else "No"}
- Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
- Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with specific recommendations available in India using INR prices.
Use Indian brands, services, and typical cost expectations.

Format your response as JSON with the following structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "venue",
      "allocation": 0.0,
      "items": [
        {{
          "name": "Community Hall Rental",
          "description": "Hall suitable for gathering",
          "estimated_price": 0.0,
          "quantity": 1,
          "search_terms": "party banquet halls nearby"
        }}
      ]
    }}
  ],
  "venue_suggestions": [
    {{
      "name": "Local Banquet Suite",
      "type": "Indoor Hall",
      "capacity": {budget_input.num_guests},
      "estimated_cost": 0.0,
      "search_terms": "banquet halls"
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": [
     "Consider potluck style for food",
     "Create a customized Spotify playlist"
  ]
}}
Ensure all costs are in INR and total does not exceed the given budget.
"""
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        result["calculation_table_inr"] = []
        categories = {}

        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "Misc")
            if cat_name not in categories:
                categories[cat_name] = {"category": cat_name, "items_count": 0, "total_cost": 0.0, "percentage_of_budget": 0.0}
            for item in category.get("items", []):
                categories[cat_name]["items_count"] += 1
                categories[cat_name]["total_cost"] += float(item.get("estimated_price", 0.0))

            if result["total_budget"] > 0:
                categories[cat_name]["percentage_of_budget"] = (categories[cat_name]["total_cost"] / result["total_budget"]) * 100

        for cat_data in categories.values():
            result["calculation_table_inr"].append(cat_data)

        category_platforms = {
            "venue": ["google", "booking", "makemytrip", "oyorooms", "nobroker"],
            "catering": ["swiggy", "zomato"],
            "food": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
            "drinks": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
            "decoration": ["amazon", "flipkart", "meesho", "myntra"],
            "entertainment": ["bookmyshow", "amazon", "flipkart"],
            "gifts": ["amazon", "flipkart", "myntra", "meesho"],
            "photography": ["google", "amazon", "flipkart"],
            "music": ["amazon", "flipkart", "bookmyshow"],
            "games": ["amazon", "flipkart"],
            "accessories": ["amazon", "flipkart", "myntra", "meesho"],
            "transportation": ["makemytrip", "google"],
            "return_gifts": ["amazon", "flipkart", "myntra", "meesho"]
        }
        default_platforms = ["amazon", "flipkart", "google"]

        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "").lower()
            relevant_platforms = category_platforms.get(cat_name, default_platforms)

            for item in category.get("items", []):
                search_terms = item.get("search_terms", "")
                if search_terms:
                    item["shopping_links"] = {}
                    q = urllib.parse.quote_plus(search_terms)
                    if "amazon" in relevant_platforms:
                        item["shopping_links"]["amazon"] = f"[https://www.amazon.in/s?k=](https://www.amazon.in/s?k=){q}"
                    if "flipkart" in relevant_platforms:
                        item["shopping_links"]["flipkart"] = f"[https://www.flipkart.com/search?q=](https://www.flipkart.com/search?q=){q}"
                    if "bigbasket" in relevant_platforms:
                        item["shopping_links"]["bigbasket"] = f"[https://www.bigbasket.com/ps/?q=](https://www.bigbasket.com/ps/?q=){q}"
                    if "swiggy" in relevant_platforms:
                        item["shopping_links"]["swiggy"] = f"[https://www.swiggy.com/search?query=](https://www.swiggy.com/search?query=){q}"
                    if "zomato" in relevant_platforms:
                        item["shopping_links"]["zomato"] = f"[https://www.zomato.com/search?q=](https://www.zomato.com/search?q=){q}"
                    if "bookmyshow" in relevant_platforms:
                        item["shopping_links"]["bookmyshow"] = f"[https://in.bookmyshow.com/search?q=](https://in.bookmyshow.com/search?q=){q}"
                    if "myntra" in relevant_platforms:
                        item["shopping_links"]["myntra"] = f"[https://www.myntra.com/search?q=](https://www.myntra.com/search?q=){q}"
                    if "meesho" in relevant_platforms:
                        item["shopping_links"]["meesho"] = f"[https://www.meesho.com/search?q=](https://www.meesho.com/search?q=){q}"
                    if "google" in relevant_platforms:
                        item["shopping_links"]["google"] = f"[https://www.google.com/search?q=](https://www.google.com/search?q=){q}"
                    if "booking" in relevant_platforms:
                        item["shopping_links"]["booking"] = f"[https://www.booking.com/search.html?ss=](https://www.booking.com/search.html?ss=){q}"
                    if "makemytrip" in relevant_platforms:
                        item["shopping_links"]["makemytrip"] = f"[https://www.makemytrip.com/hotels/hotel-listing/?searchText=](https://www.makemytrip.com/hotels/hotel-listing/?searchText=){q}"
                    if "oyorooms" in relevant_platforms:
                        item["shopping_links"]["oyorooms"] = f"[https://www.oyorooms.com/search/?location=](https://www.oyorooms.com/search/?location=){q}"
                    if "nobroker" in relevant_platforms:
                        item["shopping_links"]["nobroker"] = f"[https://www.nobroker.in/property/search?searchTerm=](https://www.nobroker.in/property/search?searchTerm=){q}"

        for venue in result.get("venue_suggestions", []):
            st = venue.get("search_terms", "")
            if st:
                q = urllib.parse.quote_plus(st)
                venue["search_links"] = {
                    "google": f"[https://www.google.com/search?q=](https://www.google.com/search?q=){q}",
                    "booking": f"[https://www.booking.com/search.html?ss=](https://www.booking.com/search.html?ss=){q}",
                    "makemytrip": f"[https://www.makemytrip.com/hotels/hotel-listing/?searchText=](https://www.makemytrip.com/hotels/hotel-listing/?searchText=){q}",
                    "oyorooms": f"[https://www.oyorooms.com/search/?location=](https://www.oyorooms.com/search/?location=){q}",
                    "nobroker": f"[https://www.nobroker.in/property/search?searchTerm=](https://www.nobroker.in/property/search?searchTerm=){q}"
                }
        return result
    except Exception as e:
        raise HTTPException(500, f"Error generating party recommendations: {str(e)}")

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    try:
        base_prompt = f"""
I need jewelry recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.
Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or "Not specified"}
Provide only India-relevant styles, availability, and price ranges in INR.
"""
        if image_path:
            img = Image.open(image_path)
            prompt = base_prompt + """
An image of the outfit is uploaded. Suggest jewelry that complements it, considering color, design, and occasion appropriateness.
Format the output as JSON:
{
  "outfit_analysis": {
    "colors": ["gold", "cream"],
    "style": "Traditional Silk",
    "formality": "Festive Wedding"
  },
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "Necklace",
      "description": "Temple jewelry antique choker",
      "style": "Traditional",
      "estimated_price": 0.0,
      "search_terms": "temple choker necklace"
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": [
    "Pair with matching jhumkas"
  ]
}
Make sure prices are in INR and stay within budget. Include Indian-friendly search terms for shopping.
"""
            response = model.generate_content([prompt, img])
        else:
            prompt = base_prompt + """
Format the output as JSON:
{
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "Earrings",
      "description": "Kundan teardrop danglers",
      "style": "Contemporary Ethnic",
      "estimated_price": 0.0,
      "search_terms": "kundan teardrop earrings"
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": [
    "Opt for minimalist accents"
  ]
}
Keep prices in INR and relevant to Indian brands.
"""
            response = model.generate_content(prompt)

        result = extract_json_from_response(response.text)

        for item in result.get("jewelry_recommendations", []):
            st = item.get("search_terms", "")
            if st:
                q = urllib.parse.quote_plus(st)
                item["shopping_links"] = {
                    "amazon": f"[https://www.amazon.in/s?k=](https://www.amazon.in/s?k=){q}",
                    "flipkart": f"[https://www.flipkart.com/search?q=](https://www.flipkart.com/search?q=){q}",
                    "bluestone": f"[https://www.bluestone.com/search.html?query=](https://www.bluestone.com/search.html?query=){q}",
                    "tanishq": f"[https://www.tanishq.co.in/search?q=](https://www.tanishq.co.in/search?q=){q}",
                    "caratlane": f"[https://www.caratlane.com/search?q=](https://www.caratlane.com/search?q=){q}",
                    "melorra": f"[https://www.melorra.com/search?q=](https://www.melorra.com/search?q=){q}",
                    "meesho": f"[https://www.meesho.com/search?q=](https://www.meesho.com/search?q=){q}"
                }
        return result
    except Exception as e:
        raise HTTPException(500, f"Error generating recommendations: {str(e)}")

# ----------------- BACKGROUND TASKS & LIFECYCLE ----------------- #
@app.on_event("startup")
async def setup_session_cleanup():
    """Background task to clean up expired sessions."""
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired_sessions = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800
            ]
            for username in expired_sessions:
                if username in active_sessions:
                    print(f"Removing expired session for {username}")
                    del active_sessions[username]
            await asyncio.sleep(300)

    asyncio.create_task(cleanup_expired_sessions())

# ----------------- AUTH & PAGE ROUTES ----------------- #
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    token = await get_token(request)
    if token:
        try:
            user = await get_current_user(request, token)
            return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        except Exception:
            pass
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    token = await get_token(request)
    if token:
        try:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        except Exception:
            pass
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    token = await get_token(request)
    if token:
        try:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        except Exception:
            pass
    return templates.TemplateResponse(request=request, name="register.html")

@app.post("/register")
async def register(user_data: RegisterUser):
    if user_data.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db[user_data.username] = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name or user_data.username,
        "hashed_password": get_password_hash(user_data.password)
    }
    return {"message": "User registered successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    existing_user_data = {}
    if user.username in active_sessions:
        existing_user_data = active_sessions[user.username].user_data
        old_token = active_sessions[user.username].token
        blacklisted_tokens.add(old_token)

    active_sessions[user.username] = UserSession(
        username=user.username,
        token=access_token,
        user_data=existing_user_data
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response

@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass

    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    history = sorted(
        user_recommendations.get(current_user.username, []),
        key=lambda x: x.timestamp,
        reverse=True
    )[:3]
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user": current_user, "history": history}
    )

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="home_planner.html",
        context={"user": current_user}
    )

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="party_planner.html",
        context={"user": current_user}
    )

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="jewelry_planner.html",
        context={"user": current_user}
    )

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"user": current_user}
    )
# ----------------- RECOMMENDATION API ENDPOINTS ----------------- #
@app.post("/home-budget")
async def plan_home_budget(budget_input: HomeBudgetInput, request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "requirements": {
                "lights": budget_input.num_lights,
                "fans": budget_input.num_fans,
                "furniture": budget_input.num_furniture,
                "dining_tables": budget_input.num_dining_tables
            }
        }
    result = get_home_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.dict(),
        result=result
    )
    return result

@app.post("/party-budget")
async def plan_party_budget(budget_input: PartyBudgetInput, request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
            "guests": budget_input.num_guests
        }
    result = get_party_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.dict(),
        result=result
    )
    return result

@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences
    )
    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image is not None
        }

    result = get_jewelry_recommendations(budget_input, image_path)

    input_data = budget_input.dict()
    if image:
        input_data["image"] = image.filename

    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result
    )
    return result

@app.get("/recommendation-history")
async def get_recommendation_history(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username not in user_recommendations:
        return {"history": []}
    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True
    )
    history_data = []
    for item in history:
        history_data.append({
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary
        })
    return {"history": history_data}

@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(recommendation_id: str, request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")
    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result
            }
    raise HTTPException(status_code=404, detail="Recommendation not found")

@app.get("/session-info")
async def get_session_info(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        session = active_sessions[current_user.username]
        return {
            "username": session.username,
            "login_time": session.login_time,
            "last_activity": session.last_activity,
            "session_duration": (datetime.utcnow() - session.login_time).total_seconds() // 60,
            "user_data": session.user_data
        }
    raise HTTPException(status_code=404, detail="No active session found")

# Entry point
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)