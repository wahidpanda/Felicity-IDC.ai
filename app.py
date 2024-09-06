from datetime import datetime
from docx import Document
import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
with st.sidebar:
  
  st.image("l.png", width=200)

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Ensure the Google API key is set
if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please add it to your .env file.")
    st.stop()

# Load custom CSS for dark theme
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# Function to check login credentials

def check_login(username, password):
    
    
    if os.path.exists("users.csv"):
        users_df = pd.read_csv("users.csv")
        if username in users_df['username'].values:
            stored_password = users_df[users_df['username'] == username]['password'].values[0]
            if stored_password == password:
                log_login(username)  # Log the login details
                return True
            else:
                st.error("Incorrect password.")
        else:
            st.error("Username not found.")
    else:
        st.error("No users registered yet.")
    return False


# Function to create a new account with date, time, and day of the week
def create_account(username, password):
    if os.path.exists("users.csv"):
        users_df = pd.read_csv("users.csv")
    else:
        # Initialize DataFrame with additional columns for date and time
        users_df = pd.DataFrame(columns=['username', 'password', 'creation_date', 'creation_time', 'creation_day'])
    
    if username in users_df['username'].values:
        st.warning("Username already exists. Please choose a different username.")
    else:
        # Get the current date, time, and day of the week
        now = datetime.now()
        creation_date = now.strftime("%Y-%m-%d")
        creation_time = now.strftime("%H:%M:%S")
        creation_day = now.strftime("%A")
        
        # Create new user DataFrame with additional columns
        new_user = pd.DataFrame({
            'username': [username],
            'password': [password],
            'creation_date': [creation_date],
            'creation_time': [creation_time],
            'creation_day': [creation_day]
        })
        
        # Append new user data to existing DataFrame
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv("users.csv", index=False)
        st.success(f"Account created successfully on {creation_day}, {creation_date} at {creation_time}!")

# Function to log user logins

def log_login(username):
    # Check if log file exists
    if os.path.exists("login_log.csv"):
        log_df = pd.read_csv("login_log.csv")
    else:
        # Initialize DataFrame for login log
        log_df = pd.DataFrame(columns=['username', 'login_date', 'login_time', 'login_day'])
    
    # Get current date, time, and day of the week
    now = datetime.now()
    login_date = now.strftime("%Y-%m-%d")
    login_time = now.strftime("%H:%M:%S")
    login_day = now.strftime("%A")
    
    # Create new log entry
    new_log = pd.DataFrame({
        'username': [username],
        'login_date': [login_date],
        'login_time': [login_time],
        'login_day': [login_day]
    })
    
    # Append new log entry to existing DataFrame
    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv("login_log.csv", index=False)

# Main app function
def main():
    # Sidebar navigation
    
    st.sidebar.title("Navigation")
    
    options = ["Advance Data Analysis", "Chat with Document", "Dashboard", "Felicity IDC ChatBot", "Advanced Analytics"]
    choice = st.sidebar.radio("Go to", options)

    if not hasattr(st.session_state, "logged_in") or not st.session_state.logged_in:
        check_login()
    else:
        show_logout_button()
    
    if choice == "Advance Data Analysis":
        st.title("Felicity IDC.ai")
        st.title("Advance Data Analysis")
        eda_page()
    elif choice == "Chat with Document":
        st.title("Chat with your Document")
        
        document_content = chat_doc()

        if document_content:
            st.text_area("Document Content", document_content, height=200)

            user_question = st.text_input("Ask your document a question:")

            if user_question:
                # Check if the vector store exists and if not, create a new one
                if not os.path.isfile("faiss_index"):
                    chunks = get_text_chunks(document_content)
                    get_vector_store(chunks)
                
                response = user_input(user_question)
                st.text_area("Response", response.get('output_text', ''), height=150)

    elif choice == "Dashboard":
        st.title("Dashboard Features")
        dashboard_page()
    elif choice == "Felicity IDC ChatBot":
        st.title("Ask any quarries, questions @ Felicity IDC ChatBot")
        chatbot_page()
    elif choice == "Advanced Analytics":
        st.title("Advanced Analytics")
        advanced_analytics_page()
    # elif choice == "Contact":
    #     st.title("Contact")
    #     st.write("Feature coming soon...")

# Function to display login page
def login_page():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.page = "main_page"
        else:
            st.sidebar.error("Incorrect username or password")
    
    st.sidebar.write("Don't have an account?")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Create Account"):
        create_account(new_username, new_password)
    
    # Main page content before login
    st.title("Felicity Internet Data Center")
    st.image("idc.jpg", caption="Welcome to Felicity Internet Data Center", use_column_width=True)
    st.title("Felicity IDC.ai")
    st.subheader("About Us")
    st.write("""
        Felicity IDC Limited (FIDC) is a Bangladesh-based ICT company that specializes in data center operations. 
        FIDC is an initiative of a few pioneering business entities, those who have already cemented their footprints in the ICT sector of Bangladesh.
        """)
    st.subheader("Key Features of Felicity IDC.ai")
    st.write("""
        - User Authentication
        - Advanced Data Analysis
        - Chat with Document
        - Custom Dashboard Creation
        - Chatbot Integration
        """)
    st.title("Contact US")
    st.image("idc2.jpeg", use_column_width=True)
    st.write("""
        Data Centre and Corporate Headquarter: Solaris (South Wing),
         """) 
    st.write("""
        Block-3, Bangabandhu Hi-Tech City (BHTC), Kaliakoir, Gazipur-1750, Bangladesh.
         """)  
    st.write("""
        Dhaka Office: Bay's 23 (Level 8 & 7), 23 Gulshan Avenue, Gulshan 1, Dhaka-1212, Bangladesh.
         """)
    st.write("""
       +8809666773355, info@felicity.net.bd 
         """)
    


# Define a function to show the logout button
def show_logout_button():
    st.sidebar.button("Logout", on_click=logout)

# Define a function to logout
def logout():
    st.session_state.logged_in = False

# Function for EDA Analysis page
def eda_page():
    # Upload CSV data
    
    with st.sidebar.header('Upload your Excel File'):
        uploaded_file = st.sidebar.file_uploader("Upload your input excel file", type=["csv"])

    # Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache_data
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.dataframe(df)  # Display the DataFrame in a tabular format
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
        
        # Custom Data Visualizations
        st.header('**Custom Data Visualizations**')
        
        # Choose plot type
        plot_type = st.selectbox("Choose a plot type", ["Histogram", "Scatter Plot", "Box Plot", "Heatmap"])

        # For Histogram
        if plot_type == "Histogram":
            column = st.selectbox("Select column for histogram", df.columns)
            bins = st.slider("Number of bins", 5, 50, 20)
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=bins)
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # For Scatter Plot
        elif plot_type == "Scatter Plot":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            fig = px.scatter(df, x=x_axis, y=y_axis)
            fig.update_layout(title=f'Scatter Plot of {x_axis} vs {y_axis}', xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig)

        # For Box Plot
        elif plot_type == "Box Plot":
            column = st.selectbox("Select column for box plot", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[column], ax=ax)
            ax.set_title(f'Box Plot of {column}')
            st.pyplot(fig)

        # For Heatmap
        elif plot_type == "Heatmap":
            st.write("Heatmap of the correlation matrix")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    else:
        st.info('Awaiting for CSV file to be uploaded.')

# Function to handle the app's state and pages
def handle_pages():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main()  # Display the main app if logged in
    else:
        login_page()  # Display the login page if not logged in
###########


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Ensure the Google API key is set
if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please add it to your .env file.")
    st.stop()

def chat_doc():
    with st.sidebar.header('Upload your File'):
        file = st.sidebar.file_uploader("Upload your input file", type=["csv", "doc", "docx", "pdf", "ppt", "pptx", "xls", "xlsx"])
    
    if file is not None:
        file_extension = file.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(file)
            return df.to_string()  # Convert DataFrame to string for processing
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
            return df.to_string()
        elif file_extension in ['doc', 'docx']:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        elif file_extension == 'pdf':
            text = get_pdf_text([file])
            return text
        elif file_extension in ['ppt', 'pptx']:
            ppt = pptx.Presentation(file)
            text = "\n".join([shape.text for slide in ppt.slides for shape in slide.shapes if hasattr(shape, "text")])
            return text
        else:
            return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Assume you are an expert consultant with in-depth knowledge in the field related to this document. Your role is to provide accurate, detailed, and helpful answers based on the information contained within the document. Please carefully analyze the content and answer the following question in a comprehensive manner, using evidence and details from the document. If the answer is not explicitly available in the document, respond by stating that the information is not available.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response


        


# Function for dashboard page
def dashboard_page():
    st.subheader("Dashboard Mode")
    
    # Allow the user to select which analysis they want to include in their custom dashboard
    st.write("Create your custom dashboard by selecting the analyses you'd like to include:")
    st.image("d.png", caption="Example Dashboard from my another project", use_column_width=True)
    # Initialize dashboard selection in session state if it doesn't exist
    if "dashboard_selection" not in st.session_state:
        st.session_state.dashboard_selection = {
            "profile": False,
            "histogram": False,
            "scatter": False,
            "boxplot": False,
            "heatmap": False
        }

    # Add options for users to choose what to include in their dashboard
    st.session_state.dashboard_selection["profile"] = st.checkbox("Include EDA Profile Report", value=st.session_state.dashboard_selection["profile"])
    st.session_state.dashboard_selection["histogram"] = st.checkbox("Include Histogram", value=st.session_state.dashboard_selection["histogram"])
    st.session_state.dashboard_selection["scatter"] = st.checkbox("Include Scatter Plot", value=st.session_state.dashboard_selection["scatter"])
    st.session_state.dashboard_selection["boxplot"] = st.checkbox("Include Box Plot", value=st.session_state.dashboard_selection["boxplot"])
    st.session_state.dashboard_selection["heatmap"] = st.checkbox("Include Heatmap", value=st.session_state.dashboard_selection["heatmap"])

    # Allow the user to save the dashboard configuration
    if st.button("Save Dashboard"):
        st.success("Dashboard saved!")

    # Display dashboard based on user selection
    if st.session_state.dashboard_selection["profile"]:
        st.subheader("EDA Profile Report")
        st.write("Placeholder for the EDA Profile Report")
        # Here you'd include the actual profile report or code to display it
        
    if st.session_state.dashboard_selection["histogram"]:
        st.subheader("Histogram")
        st.write("Placeholder for Histogram")
        # Code for histogram goes here
        
    if st.session_state.dashboard_selection["scatter"]:
        st.subheader("Scatter Plot")
        st.write("Placeholder for Scatter Plot")
        # Code for scatter plot goes here
        
    if st.session_state.dashboard_selection["boxplot"]:
        st.subheader("Box Plot")
        st.write("Placeholder for Box Plot")
        # Code for box plot goes here
        
    if st.session_state.dashboard_selection["heatmap"]:
        st.subheader("Heatmap")
        st.write("Placeholder for Heatmap")
        # Code for heatmap goes here

    st.write("Feature coming soon: Save and load custom dashboards.")


#chatbot intrigration here

def chatbot_page():
    load_dotenv()

# Configure Streamlit page settings
    

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')


    # Function to translate roles between Gemini-Pro and Streamlit terminology
    def translate_role_for_streamlit(user_role):
        if user_role == "model":
            return "assistant"
        else:
            return user_role


    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])


    # Display the chatbot's title on the page
    

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask me Q relaed Data Center....")
   
    if user_prompt:
        
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Advanced Analytics
def advanced_analytics_page():
    

    # Predictive Modeling
    st.subheader("Predictive Modeling")
    model_option = st.selectbox("Select a predictive model", ["None", "Decision Tree"])
    
    if model_option == "Decision Tree":
        st.write("You selected: Decision Tree")

        # Upload CSV data
        uploaded_file = st.file_uploader("Upload your CSV file for modeling", type=["csv"])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Select features and target
            features = st.multiselect("Select feature columns", df.columns)
            target = st.selectbox("Select target column", df.columns)
            
            if features and target:
                # Split data
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Show progress bar while training
                with st.spinner("Training the model..."):
                    st.progress(0.5)  # Show 50% progress
                    model = DecisionTreeClassifier()
                    model.fit(X_train, y_train)
                    accuracy = model.score(X_test, y_test)
                    st.progress(1.0)  # Show 100% progress
                    
                st.write(f"Model accuracy: {accuracy:.2f}")
            else:
                st.warning("Please select feature columns and target column.")

    # Advanced Statistical Analysis
    st.subheader("Advanced Statistical Analysis")
    analysis_option = st.selectbox("Select statistical analysis", ["None", "Hypothesis Testing"])
    
    if analysis_option == "Hypothesis Testing":
        st.write("You selected: Hypothesis Testing")
        
        # Upload CSV data
        uploaded_file = st.file_uploader("Upload your CSV file for statistical analysis", type=["csv"])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Select columns for hypothesis testing
            column1 = st.selectbox("Select first column", df.columns)
            column2 = st.selectbox("Select second column", df.columns)
            
            if column1 and column2:
                # Perform hypothesis testing
                with st.spinner("Performing hypothesis testing..."):
                    st.progress(0.5)  # Show 50% progress
                    t_statistic, p_value = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
                    st.progress(1.0)  # Show 100% progress
                    
                st.write(f"T-statistic: {t_statistic:.2f}")
                st.write(f"P-value: {p_value:.2f}")
            else:
                st.warning("Please select both columns for analysis.")

if __name__ == "__main__":
    handle_pages()
