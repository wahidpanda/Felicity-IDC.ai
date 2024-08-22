# Felicity-IDC.ai
Felicity IDC.ai is a web application for data analysis and management at Felicity Internet Data Center, utilizing AI, chatbots, custom dashboards, and advanced analytics tools.


![image](https://github.com/user-attachments/assets/dad5cd1e-7b40-45c0-b90d-348f9fac1226)

# Felicity IDC.ai

Felicity IDC.ai is an advanced data analysis and interactive application for Felicity Internet Data Center (FIDC). It integrates various features including data analysis, chatbot interaction, custom dashboards, and advanced analytics.

## Project Structure

Felicity-IDC.ai/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── eda_page.py
│   │   ├── chat_pdf.py
│   │   ├── dashboard_page.py
│   │   ├── login_page.py
│   │   ├── chatbot_page.py
│   │   ├── advanced_analytics_page.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── pdf_processing.py
│   │   ├── gtts_speech.py
│   │   ├── vector_store.py
│   │   ├── custom_plots.py
├── static/
│   ├── css/
│   │   └── custom_style.css
│   ├── images/
│       ├── idc.jpg
│       ├── idc2.jpeg
├── .env.example
└── LICENSE

## Features

- **User Authentication**: Secure login and account creation system.
- **Advanced Data Analysis**: Upload and analyze datasets with custom visualizations.
- **Chat with Document**: Interact with PDF documents using Google's Gemini API.
- **Custom Dashboard Creation**: Build and save dashboards based on your data analysis preferences.
- **Chatbot Integration**: Query Felicity IDC's systems using an AI-powered chatbot.
- **Advanced Analytics**: Perform predictive modeling and advanced statistical analysis.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/wahidpanda/Felicity-IDC.ai.git
    cd Felicity-IDC.ai
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your environment variables by copying `.env.example` to `.env` and updating with your credentials:
    ```sh
    cp .env.example .env
    ```

5. Run the application:
    ```sh
    streamlit run app/main.py
    ```

## Dependencies

- `Streamlit`: Web framework for creating interactive applications.
- `pandas`, `ydata-profiling`: Data manipulation and analysis.
- `Google Generative AI`: API for interacting with PDF documents.
- `FAISS`: Vector search engine for text embeddings.
- `PyPDF2`: PDF processing library.
- `gTTS`: Text-to-speech conversion.
- `seaborn`, `matplotlib`, `plotly`: Data visualization libraries.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contribution

This is not for contribuiton purpose !!!

## COntact
Email me: islamoahidul12@gmail.com 


