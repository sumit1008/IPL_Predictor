import streamlit as st
import pickle
import pandas as pd

# Load model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Team and city options
teams = ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
        'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
        'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
        'Sunrisers Hyderabad']

cities = ['Chandigarh', 'Delhi', 'Kolkata', 'Jaipur', 'Hyderabad', 'Chennai',
        'Mumbai', 'Cape Town', 'Durban', 'Port Elizabeth', 'Centurion',
        'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
        'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam',
        'Pune', 'Bangalore', 'Raipur', 'Abu Dhabi', 'Ranchi', 'Indore',
        'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati',
        'Mohali', 'Bengaluru']

# App title
st.markdown("<h1 style='text-align: center; color: orange;'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)

# Sidebar for match setup
st.sidebar.header("Match Setup")

batting_team = st.sidebar.selectbox('Select Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Select Bowling Team', sorted([t for t in teams if t != batting_team]))
selected_city = st.sidebar.selectbox('Select Match City', sorted(cities))
target = st.sidebar.number_input('Target Score', min_value=1, step=1)

# Match progress section
st.markdown("### Live Match Input")

col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input('Current Score', min_value=0, step=1)
with col2:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
with col3:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('🔮 Predict Win Probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'cur_rr': [crr],
        'req_rr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Determine colors based on who's winning
    if abs(win - loss) < 0.01:
        batting_color = "white"
        bowling_color = "white"
        font_color = "black"
    else:
        batting_color = "green" if win > loss else "red"
        bowling_color = "green" if loss > win else "red"
        font_color = "white"

    # Display probabilities with colored boxes
    st.markdown("## 🧮 Win Probability")

    st.markdown(f"""
    <div style='background-color:{batting_color};padding:10px;border-radius:10px;margin-bottom:10px;'>
        <h4 style='color:{font_color};text-align:center;'>{batting_team} Win Probability: {round(win * 100)}%</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background-color:{bowling_color};padding:10px;border-radius:10px;'>
        <h4 style='color:{font_color};text-align:center;'>{bowling_team} Win Probability: {round(loss * 100)}%</h4>
    </div>
    """, unsafe_allow_html=True)

    # Optional: Visual bar (green/red)
    st.markdown("### 📈 Probability Bar")
    st.progress(win)
    # st.table(input_df)
    st.markdown("---")
    st.write("📊 *Model prediction based on current match status.*")
