import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import requests
from urllib.parse import quote
import time
import uuid
from datetime import datetime


# === Page Configuration ===
st.set_page_config(
    page_title="🛍️ Friendly Store Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Load Environment Variables ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in environment variables!")
    st.error("Please create a .env file with your OpenAI API key")
    st.stop()

# === Initialize OpenAI Client ===
try:
    client = OpenAI(api_key=openai_api_key)
    # Test connection
    client.models.list()
    st.success("✅ OpenAI API connected successfully")
except Exception as e:
    st.error(f"❌ OpenAI API connection failed: {str(e)}")
    st.stop()

# === Load Inventory Database ===
@st.cache_data
def load_inventory():
    """Load inventory from CSV file with proper error handling"""
    try:
        path = "store_db/DMart.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Clean column names
            df.columns = [col.lower().strip() for col in df.columns]
            return df
        else:
            st.warning("⚠️ Inventory file not found at store_db/DMart.csv")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading inventory: {str(e)}")
        return pd.DataFrame()

# === Enhanced Product Categorization ===
def categorize_product(row):
    """Categorize product based on available information"""
    # Create searchable text excluding description to avoid confusion
    searchable_columns = ['name', 'brand', 'category', 'type', 'item']
    product_text = ""
    
    for col in searchable_columns:
        if col in row and pd.notna(row[col]):
            product_text += f" {str(row[col]).lower()}"
    
    # Define category keywords (more specific matching)
    categories = {
        'personal_care': {
            'perfume': ['perfume', 'fragrance', 'cologne', 'eau de parfum', 'deodorant', 'body spray'],
            'skincare': ['cream', 'lotion', 'moisturizer', 'face wash', 'cleanser', 'sunscreen'],
            'haircare': ['shampoo', 'conditioner', 'hair oil', 'hair cream', 'hair gel'],
            'oral_care': ['toothpaste', 'toothbrush', 'mouthwash']
        },
        'food_beverages': {
            'snacks': ['chips', 'biscuit', 'cookie', 'wafer', 'namkeen'],
            'beverages': ['juice', 'drink', 'water', 'soda', 'tea', 'coffee'],
            'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'curd']
        },
        'household': {
            'cleaning': ['detergent', 'soap', 'cleaner', 'disinfectant'],
            'fabric_care': ['fabric conditioner', 'fabric softener', 'softener', 'softouch']
        }
    }
    
    # Find the most specific category
    for main_cat, sub_cats in categories.items():
        for sub_cat, keywords in sub_cats.items():
            for keyword in keywords:
                if keyword in product_text:
                    return f"{main_cat}_{sub_cat}"
    
    return "general"

# === Enhanced Inventory Search Functions ===
def search_inventory(query, df, max_results=10):
    """
    Improved inventory search - finds products by name, brand, type
    Fixes the rice search issue by removing restrictive categorization penalties
    """
    if df.empty:
        return []
    
    query_lower = query.lower().strip()
    keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 1]
    
    matches = []
    
    for idx, row in df.iterrows():
        # Create searchable text from ALL relevant columns
        search_columns = ['name', 'brand', 'category', 'type', 'item', 'product', 'title', 'description']
        searchable_text = ""
        
        for col in search_columns:
            if col in df.columns and pd.notna(row.get(col)):
                searchable_text += f" {str(row[col]).lower()}"
        
        # Calculate match score - SIMPLIFIED AND MORE INCLUSIVE
        match_score = 0
        
        # 1. Direct substring match (highest priority)
        if query_lower in searchable_text:
            match_score += 50
        
        # 2. Individual keyword matches
        for keyword in keywords:
            if keyword in searchable_text:
                match_score += 20
                # Extra points for exact matches in important fields
                for important_col in ['name', 'product', 'item']:
                    if important_col in df.columns and pd.notna(row.get(important_col)):
                        if keyword in str(row[important_col]).lower():
                            match_score += 30
        
        # 3. Partial word matching (for variants like "basmati rice", "brown rice")
        for keyword in keywords:
            for word in searchable_text.split():
                if keyword in word and len(keyword) > 2:
                    match_score += 10
        
        # 4. Brand matching bonus
        if 'brand' in df.columns and pd.notna(row.get('brand')):
            brand_text = str(row['brand']).lower()
            for keyword in keywords:
                if keyword in brand_text:
                    match_score += 15
        
        # Add to matches if any score
        if match_score > 0:
            row_dict = row.to_dict()
            row_dict['match_score'] = match_score
            matches.append(row_dict)
    
    # Sort by match score and return top results
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    return matches[:max_results]


def get_query_intent(query):
    """Determine the intent/category of the user's query"""
    intent_keywords = {
        'perfume': ['perfume', 'fragrance', 'cologne', 'scent'],
        'skincare': ['cream', 'lotion', 'face wash', 'skincare'],
        'haircare': ['shampoo', 'conditioner', 'hair'],
        'snacks': ['snacks', 'chips', 'biscuit', 'cookie'],
        'beverages': ['drink', 'juice', 'water', 'beverage'],
        'cleaning': ['detergent', 'soap', 'cleaner']
    }
    
    for intent, keywords in intent_keywords.items():
        if any(keyword in query for keyword in keywords):
            return intent
    return None

def get_enhanced_product_summary(products):
   """Convert product list to detailed readable summary with better information"""
   if not products:
       return "No products found in our inventory."
   
   summary = []
   for product in products[:8]:  # Show more products (was 5)
       # Extract key information with better field mapping
       name = "Unknown Product"
       price = "Price not available"
       brand = "Unknown Brand"
       category = product.get('category', 'general')
       
       # Try to extract product name - check more fields
       for key in ['name', 'product', 'item', 'title', 'product_name']:
           if key in product and pd.notna(product[key]):
               name = str(product[key]).strip()
               break
       
       # IMPROVED PRICE EXTRACTION - handles multiple price fields and discounts
       price_found = False
       # Check for discounted/sale prices first (customer pays less)
       for key in ['DiscountedPrice', 'sale_price', 'offer_price', 'final_price', 'selling_price']:
           if key in product and pd.notna(product[key]):
               try:
                   price_val = float(str(product[key]).replace('₹', '').replace(',', ''))
                   if price_val > 0:
                       price = f"₹{price_val:.2f}"
                       price_found = True
                       break
               except (ValueError, TypeError):
                   continue
       
       # If no discounted price, check regular prices
       if not price_found:
           for key in ['price', 'mrp', 'cost', 'amount', 'rate']:
               if key in product and pd.notna(product[key]):
                   try:
                       price_val = float(str(product[key]).replace('₹', '').replace(',', ''))
                       if price_val > 0:
                           price = f"₹{price_val:.2f}"
                           price_found = True
                           break
                   except (ValueError, TypeError):
                       # If conversion fails, try to use as string
                       price = f"₹{str(product[key]).strip()}"
                       price_found = True
                       break
       
       # Try to extract brand
       for key in ['brand', 'manufacturer', 'company', 'brand_name']:
           if key in product and pd.notna(product[key]):
               brand = str(product[key]).strip()
               break
       
       # Add additional details if available - more comprehensive
       details = []
       detail_fields = ['size', 'weight', 'quantity', 'pack', 'volume', 'net_weight', 'pack_size', 'unit']
       for key in detail_fields:
           if key in product and pd.notna(product[key]):
               value = str(product[key]).strip()
               if value and value.lower() not in ['nan', 'none', '']:
                   details.append(f"{key.replace('_', ' ').title()}: {value}")
       
       # Format the summary line
       detail_str = f" ({', '.join(details)})" if details else ""
       
       # Clean up brand display
       brand_display = f" by {brand}" if brand != "Unknown Brand" else ""
       
       # Clean up category display
       category_display = category.replace('_', ' ').title() if category != 'general' else ""
       category_str = f" [Category: {category_display}]" if category_display else ""
       
       summary.append(f"• {name}{brand_display} - {price}{detail_str}{category_str}")
   
   return "\n".join(summary)

# === Bill Calculation Function ===
def calculate_bill(products_with_quantities):
    """
    Calculate total bill - handles various price field names
    """
    total = 0.0
    calculation_details = []
    
    for item in products_with_quantities:
        try:
            # Get product name
            product_name = "Unknown Product"
            for name_field in ['name', 'product', 'item', 'title']:
                if name_field in item and pd.notna(item[name_field]):
                    product_name = str(item[name_field])
                    break
            
            # Get quantity
            quantity = float(item.get('quantity', 1))
            
            # Get price - try multiple fields
            price = 0.0
            for price_field in ['price', 'cost', 'amount', 'mrp', 'rate']:
                if price_field in item and pd.notna(item[price_field]):
                    try:
                        price_str = str(item[price_field]).replace('₹', '').replace(',', '')
                        price = float(price_str)
                        break
                    except (ValueError, TypeError):
                        continue
            
            item_total = price * quantity
            total += item_total
            
            calculation_details.append({
                'name': product_name,
                'price': price,
                'quantity': quantity,
                'total': item_total
            })
            
        except Exception as e:
            calculation_details.append({
                'name': item.get('name', 'Unknown'),
                'error': f"Could not calculate: {str(e)}"
            })
    
    return {
        'total': total,
        'details': calculation_details,
        'formatted_total': f"₹{total:.2f}"
    }

def should_search_inventory(query):
    """Detect if query needs inventory search"""
    shopping_triggers = [
        'buy', 'need', 'want', 'looking for', 'show me', 'find',
        'bulking', 'skincare', 'cleaning', 'cooking', 'snacks',
        'products for', 'items for', 'what should i buy',
        'supplements', 'protein', 'food', 'cosmetics', 'medicine'
    ]
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in shopping_triggers)
# === Internet Search Function ===

def search_internet(query, max_results=3):
    """Search internet for additional product information"""
    try:
        # Try to import duckduckgo-search
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for result in search_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'summary': result.get('body', 'No summary'),
                    'url': result.get('href', ''),
                })
        
        return results
    except ImportError:
        st.warning("⚠️ DuckDuckGo search not available. Install with: pip install duckduckgo-search")
        return []
    except Exception as e:
        st.warning(f"⚠️ Internet search failed: {str(e)}")
        return []

# === Enhanced OpenAI Assistant Management ===
class FriendlyStoreAssistant:
    def __init__(self, client, inventory_df):
        self.client = client
        self.inventory_df = inventory_df
        self.assistant_id = None
        self.thread_id = None
        
        # Create or get assistant
        self._setup_assistant()
    
    def _setup_assistant(self):
        """Create the enhanced friendly store assistant"""
        
        from datetime import datetime

        # Get current date for context
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")
        
        # FIXED: Priority-driven assistant instructions
        instructions = f"""    You are Aiko, a friendly and intelligent store assistant! 🛍️ 

    IMPORTANT CONTEXT:
    - Current date: {current_date}
    - Current time: {current_time}
    - You work at a physical store with a comprehensive inventory database

    🎯 **CRITICAL PRIORITY RULES** (Follow in this exact order):

    **RULE #1: INVENTORY FIRST POLICY**
    - - You are EXCLUSIVELY a store/shopping assistant
    - For ANY product-related question, ALWAYS search store inventory FIRST
    - Examples: "what should I buy for bulking?" → Search for proteins, supplements, healthy foods
    - Examples: "I need skincare products" → Search inventory for creams, lotions, face wash
    - Examples: "show me snacks" → Search inventory for chips, biscuits, cookies
    - NEVER give generic product advice without checking what's actually in stock

    **RULE #2: ADAPTIVE QUERY UNDERSTANDING**
    - Use AI to understand the TRUE intent behind any query
    - Don't rely on keyword matching - understand context
    - Examples:
    * "I want something for dry skin" → Search skincare products
    * "Need energy for workout" → Search supplements/energy products  
    * "My hair is falling" → Search hair care products
    * "Party tonight, need to smell good" → Search perfumes
    - ALWAYS ask yourself: "What is the customer actually trying to solve?"

    **RULE #3: RESPONSE PATTERN FOR PRODUCT QUERIES**
    1. FIRST: Search inventory with relevant keywords
    2. SECOND: Present available products with prices
    3. THIRD: If inventory is limited, then provide general guidance
    4. FOURTH: Suggest alternatives from inventory

    **RULE #4: BILL CALCULATION PRIORITY**
    - ALWAYS use calculate_bill function for any price calculations
    - Never do manual math for totals

    **RULE #5: CURRENT INFO PRIORITY**
    - Only search internet for time-sensitive queries (news, sports, weather)
    - For product information, prioritize inventory over internet search

    EXAMPLE INTERACTION FLOW:
    User: "I'm bulking, what should I buy?"

    Your Response Process:
    1. ✅ Search inventory: "protein supplements bulking foods"
    2. ✅ Search inventory: "eggs milk chicken rice oats"  
    3. ✅ Present: "Here's what we have in stock for bulking..."
    4. ✅ If limited stock: "We also recommend these general bulking foods..."

    ❌ WRONG: Starting with generic bulking advice
    ✅ RIGHT: Starting with "Let me check what bulking products we have in stock..."

    RESPONSE STYLE:
    - Always lead with inventory search results
    - Be specific about what's available vs. not available
    - Show prices and details from your store
    - Use friendly tone with emojis 😊
    - add freindly puns where you fill its funny.
    """

        # Define the function tools for the assistant
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_store_inventory",
                    "description": "Search the store's inventory database for products. Use this for ANY product-related query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for products (e.g., 'shampoo', 'protein supplements', 'skincare cream')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of products to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_bill",
                    "description": "Calculate total bill for selected products with quantities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "products_with_quantities": {
                                "type": "array",
                                "description": "List of products with their quantities",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "price": {"type": "number"},
                                        "quantity": {"type": "number"}
                                    }
                                }
                            }
                        },
                        "required": ["products_with_quantities"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_internet_info",
                    "description": "Search internet for current information like news, weather, sports updates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for current information"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        try:
            # Create the assistant
            assistant = self.client.beta.assistants.create(
                name="Aiko - Enhanced Store Assistant",
                instructions=instructions,
                model="gpt-4o-mini",  # Use the latest model
                tools=tools
            )
            
            self.assistant_id = assistant.id
            st.success(f"✅ Assistant created successfully! ID: {self.assistant_id[:10]}...")
            
        except Exception as e:
            st.error(f"❌ Failed to create assistant: {str(e)}")
            raise e

    def create_thread(self):
        """Create a new conversation thread"""
        try:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            return thread.id
        except Exception as e:
            st.error(f"❌ Failed to create thread: {str(e)}")
            return None
    
    def send_message(self, message):
        """Send message to assistant and get response with inventory priority"""
        if not self.thread_id:
            self.create_thread()
        
        try:
            # 🟢 NEW: Auto-detect inventory search need
            original_message = message
            if should_search_inventory(message):
                # Add context to prioritize inventory search
                message = f"""
CUSTOMER QUERY: {original_message}

PRIORITY INSTRUCTION: This appears to be a product/shopping query. You MUST search the store inventory FIRST using the search_store_inventory function before providing any response. Show what products we actually have in stock with prices and details.

Only provide general advice if our inventory doesn't have relevant products.
"""
            
            # Add message to thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion and handle function calls
            return self._wait_for_run_completion(run.id)
            
        except Exception as e:
            st.error(f"❌ Failed to send message: {str(e)}")
            return "Sorry, I encountered an error. Please try again! 😅"
    def _wait_for_run_completion(self, run_id):
        """Wait for run completion and handle function calls"""
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            try:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run_id
                )
                
                if run.status == "completed":
                    # Get the latest message
                    messages = self.client.beta.threads.messages.list(
                        thread_id=self.thread_id
                    )
                    
                    latest_message = messages.data[0]
                    return latest_message.content[0].text.value
                
                elif run.status == "requires_action":
                    # Handle function calls
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == "search_store_inventory":
                            # Enhanced inventory search
                            query = function_args.get("query", "")
                            max_results = function_args.get("max_results", 5)
                            print(query)
                            products = search_inventory(query, self.inventory_df, max_results)
                            result = {
                                "products_found": len(products),
                                "products": products,
                                "summary": get_enhanced_product_summary(products),
                                "search_query": query
                            }
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(result)
                            })
                        
                        elif function_name == "calculate_bill":
                            # Enhanced bill calculation
                            products_with_quantities = function_args.get("products_with_quantities", [])
                            
                            bill_result = calculate_bill(products_with_quantities)
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(bill_result)
                            })
                        
                        elif function_name == "search_internet_info":
                            # Internet search for current information
                            query = function_args.get("query", "")
                            internet_results = search_internet(query)
                            
                            result = {
                                "results_found": len(internet_results),
                                "results": internet_results,
                                "search_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(result)
                            })
                    
                    # Submit tool outputs
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread_id,
                        run_id=run_id,
                        tool_outputs=tool_outputs
                    )
                
                elif run.status in ["failed", "cancelled", "expired"]:
                    return f"Sorry, something went wrong with my processing. Status: {run.status} 😅"
                
                # Wait a bit before next check
                time.sleep(1)
                attempt += 1
                
            except Exception as e:
                st.error(f"❌ Error in run completion: {str(e)}")
                return "Sorry, I encountered an error while processing your request! 😅"
        
        return "Sorry, I'm taking too long to respond. Please try again! ⏰"

# === Streamlit UI ===

st.title("🛍️ Enhanced Friendly Store Assistant")
st.caption("Hi! I'm Aiko, your enhanced store assistant! I can help with accurate product searches, bill calculations, and current information! 😊")

# === Initialize Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "assistant" not in st.session_state:
    # Load inventory
    inventory_df = load_inventory()
    
    # Create assistant
    with st.spinner("🤖 Setting up your enhanced assistant..."):
        try:
            st.session_state.assistant = FriendlyStoreAssistant(client, inventory_df)
            st.session_state.thread_id = st.session_state.assistant.create_thread()
        except Exception as e:
            st.error(f"❌ Failed to initialize assistant: {str(e)}")
            st.stop()

# === Enhanced Sidebar ===
with st.sidebar:
    st.header("🏪 Enhanced Store Status")
    
    # Current time display
    st.info(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Inventory status
    inventory_df = load_inventory()
    if not inventory_df.empty:
        st.success(f"📦 {len(inventory_df)} products available")
        
        # Enhanced sample products with categories
        with st.expander("Sample Products by Category"):
            # Group by category
            sample_products = inventory_df.head(10)
            for _, row in sample_products.iterrows():
                category = categorize_product(row)
                
                # Find product name
                product_name = "Unknown Product"
                for col in ['name', 'product', 'item', 'title']:
                    if col in row and pd.notna(row[col]):
                        product_name = str(row[col])[:30] + "..." if len(str(row[col])) > 30 else str(row[col])
                        break
                
                # Find price
                price = ""
                for col in ['price', 'cost', 'mrp']:
                    if col in row and pd.notna(row[col]):
                        price = f" - ₹{row[col]}"
                        break
                
                st.write(f"🏷️ **{category.replace('_', ' ').title()}**")
                st.write(f"   • {product_name}{price}")
    else:
        st.warning("📦 No inventory loaded")
    
    # Enhanced feature status
    try:
        from duckduckgo_search import DDGS
        st.success("🔍 Internet search: ✅ Active")
    except ImportError:
        st.warning("🔍 Internet search: ⚠️ Limited")
    
    st.divider()
    
    st.header("💬 Enhanced Examples")
    st.code("Calculate my bill for 2 shampoos and 3 soaps")
    st.code("Who won the latest IPL match?")
    st.code("Show me actual perfumes, not fabric conditioners")
    st.code("What's the weather today in Bangalore?")
    st.code("Find me skincare products under ₹500")
    
    st.divider()
    
    st.header("🎯 Enhanced Features")
    st.write("🧮 **Accurate Bill Calculation**")
    st.write("🎯 **Smart Product Categorization**")
    st.write("🌐 **Current Information Search**")
    st.write("📊 **Detailed Product Analysis**")
    st.write("🤖 **Context-Aware Responses**")
    st.write("🔍 **Enhanced Search Algorithm**")
    
    if st.button("🗑️ Clear Chat & Start Fresh"):
        st.session_state.chat_history = []
        # Create new thread
        st.session_state.thread_id = st.session_state.assistant.create_thread()
        st.rerun()

# === Main Chat Interface ===
col1, col2 = st.columns([4, 1])

with col1:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    # Chat input
    user_input = st.chat_input("Ask me anything! I'm now more accurate and helpful! 😊")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking carefully..."):
                try:
                    response = st.session_state.assistant.send_message(user_input)
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.chat_history.append(("assistant", response))
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)} 😅 Please try again!"
                    st.markdown(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))

with col2:
    st.header("🔧 Quick Actions")
    
    # Test bill calculation
    if st.button("🧮 Test Bill Calculation"):
        test_query = "Calculate my bill for 2 items at ₹100 each and 1 item at ₹50"
        st.session_state.chat_history.append(("user", test_query))
        with st.spinner("Calculating..."):
            response = st.session_state.assistant.send_message(test_query)
            st.session_state.chat_history.append(("assistant", response))
        st.rerun()
    
    # Test current info
    if st.button("📰 Get Current News"):
        news_query = "What's the latest news today?"
        st.session_state.chat_history.append(("user", news_query))
        with st.spinner("Fetching current news..."):
            response = st.session_state.assistant.send_message(news_query)
            st.session_state.chat_history.append(("assistant", response))
        st.rerun()
    
    # Test product categorization
    if st.button("🏷️ Show Product Categories"):
        cat_query = "Show me different types of products you have, organized by category"
        st.session_state.chat_history.append(("user", cat_query))
        with st.spinner("Organizing products..."):
            response = st.session_state.assistant.send_message(cat_query)
            st.session_state.chat_history.append(("assistant", response))
        st.rerun()

# === Footer ===
st.divider()
st.caption("💡 **Enhanced Features**: Accurate calculations, smart categorization, current information, and better product matching! 🚀")

# === Setup Instructions ===
if inventory_df.empty:
    st.warning("⚠️ **Setup Required**: Place your inventory CSV file at `store_db/DMart.csv`")
    st.info("📋 **CSV Format**: Should include columns like 'name', 'brand', 'price', 'category', etc.")