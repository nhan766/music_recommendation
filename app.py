import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sqlite3
import hashlib
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH TRANG WEB & CSS
# ==========================================
st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .music-card { background: linear-gradient(145deg, #1e222b, #15181f); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2d3139; }
    .music-title { color: #ffffff; font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    .music-artist { color: #b3b3b3; font-size: 14px; margin-bottom: 10px; }
    .music-badge { background-color: #282c34; color: #1db954; padding: 4px 10px; border-radius: 20px; font-size: 11px; }
    .sim-bar-container { background-color: #282c34; border-radius: 10px; height: 6px; width: 100%; margin-top: 10px; }
    .sim-bar-fill { background: linear-gradient(90deg, #1db954, #1ed760); height: 6px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU (SQLite)
# ==========================================
# Hàm băm mật khẩu để bảo mật
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Khởi tạo Database và các Bảng
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable (username TEXT, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS historytable (username TEXT, song_title TEXT, search_time TEXT)')
    conn.commit()
    return conn

# Các hàm thao tác dữ liệu
def add_user(username, password):
    conn = init_db()
    c = conn.cursor()
    c.execute('INSERT INTO userstable (username, password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    conn = init_db()
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', (username, password))
    return c.fetchall()

def add_search_history(username, song_title):
    conn = init_db()
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO historytable (username, song_title, search_time) VALUES (?,?,?)', (username, song_title, now))
    conn.commit()

def get_user_history(username):
    conn = init_db()
    c = conn.cursor()
    c.execute('SELECT song_title, search_time FROM historytable WHERE username = ? ORDER BY search_time DESC LIMIT 10', (username,))
    return c.fetchall()

init_db() # Gọi hàm để tạo file DB ngay khi chạy app

# ==========================================
# 3. LOGIC HỆ THỐNG GỢI Ý (Giữ nguyên)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        if 'popularity' not in df.columns:
            np.random.seed(42)
            df['popularity'] = np.random.randint(10, 100, size=len(df))
        df = df[['track_name', 'artists', 'track_genre', 'popularity']].dropna()
        df = df.drop_duplicates(subset=['track_name'])
        return df.sample(n=5000, random_state=42).reset_index(drop=True)
    except:
        data = {
            'track_name': ['Shape of You', 'Bohemian Rhapsody', 'Perfect'],
            'artists': ['Ed Sheeran', 'Queen', 'Ed Sheeran'],
            'track_genre': ['Pop', 'Rock', 'Pop'],
            'popularity': [95, 90, 88]
        }
        return pd.DataFrame(data)

df = load_data()

def hybrid_recommend(song_title, df, top_n=4, content_weight=0.7, pop_weight=0.3):
    df['Features'] = df['track_genre'] + " " + df['artists']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    normalized_pop = df['popularity'] / 100.0
    idx = df[df['track_name'] == song_title].index[0]
    
    hybrid_scores = []
    for i, content_score in enumerate(cosine_sim[idx]):
        if i != idx:
            final_score = (content_weight * content_score) + (pop_weight * normalized_pop.iloc[i])
            hybrid_scores.append((i, final_score))
            
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for i, score in hybrid_scores:
        row = df.iloc[i].copy()
        row['hybrid_score'] = score
        results.append(row)
    return pd.DataFrame(results)

# ==========================================
# 4. QUẢN LÝ PHIÊN (Session State) & GIAO DIỆN
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# NẾU CHƯA ĐĂNG NHẬP -> Hiển thị form Đăng nhập/Đăng ký
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>🎵 Welcome to Music Recommender</h1>", unsafe_allow_html=True)
    
    menu = ["Đăng nhập", "Đăng ký"]
    choice = st.selectbox("Chọn thao tác", menu)

    if choice == "Đăng nhập":
        st.subheader("Đăng nhập vào hệ thống")
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type='password')
        if st.button("Đăng nhập"):
            hashed_pswd = make_hashes(password)
            result = login_user(username, hashed_pswd)
            if result:
                st.success(f"Đăng nhập thành công! Chào mừng {username}")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.rerun() # Tải lại trang để vào hệ thống chính
            else:
                st.warning("Tên đăng nhập hoặc mật khẩu không đúng!")

    elif choice == "Đăng ký":
        st.subheader("Tạo tài khoản mới")
        new_user = st.text_input("Tên đăng nhập mới")
        new_password = st.text_input("Mật khẩu mới", type='password')
        if st.button("Đăng ký"):
            add_user(new_user, make_hashes(new_password))
            st.success("Tạo tài khoản thành công! Bạn có thể chuyển sang mục Đăng nhập.")

# NẾU ĐÃ ĐĂNG NHẬP -> Hiển thị Hệ thống Gợi ý Âm nhạc
else:
    # Sidebar quản lý tài khoản và Lịch sử
    with st.sidebar:
        st.write(f"👤 Xin chào, **{st.session_state['username']}**")
        if st.button("Đăng xuất"):
            st.session_state['logged_in'] = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🕒 Lịch sử tìm kiếm của bạn")
        history_data = get_user_history(st.session_state['username'])
        if history_data:
            for song, time_str in history_data:
                st.markdown(f"- **{song}** <br><span style='font-size: 10px; color: #888;'>{time_str}</span>", unsafe_allow_html=True)
        else:
            st.write("Chưa có lịch sử.")

    # Giao diện Hệ thống chính
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>🎵 Hybrid Music Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b3b3b3;'>Course: Recommendation Systems | Instructor: PhD. Nguyen Luong Vuong</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎧 Khám phá Âm nhạc", "📊 Đánh giá Mô hình"])

    with tab1:
        col_input, col_spacer, col_output = st.columns([1.5, 0.2, 3])
        with col_input:
            song_list = df['track_name'].tolist()
            selected_song = st.selectbox("🔍 Nhập tên bài hát:", options=song_list, index=None)
            
            content_weight = st.slider("Tỷ trọng Lọc nội dung (Content-based)", 0.0, 1.0, 0.7)
            pop_weight = 1.0 - content_weight
            
            search_button = st.button("Phân tích & Gợi ý", use_container_width=True, disabled=(selected_song is None))

        with col_output:
            if search_button and selected_song:
                # Ghi lịch sử vào Database ngay khi người dùng bấm tìm kiếm
                add_search_history(st.session_state['username'], selected_song)
                
                st.markdown(f"<h3 style='color: #ffffff;'>✨ Gợi ý Top-N cho: {selected_song}</h3>", unsafe_allow_html=True)
                with st.spinner("Đang tính toán ma trận..."):
                    recommendations = hybrid_recommend(selected_song, df, top_n=4, content_weight=content_weight, pop_weight=pop_weight)
                    
                    grid_col1, grid_col2 = st.columns(2)
                    for index, row in recommendations.iterrows():
                        target_col = grid_col1 if index % 2 == 0 else grid_col2
                        sim_percentage = min(int(row['hybrid_score'] * 100), 100) 
                        
                        with target_col:
                            st.markdown(f"""
                                <div class="music-card">
                                    <div class="music-title">🎵 {row['track_name']}</div>
                                    <div class="music-artist">👤 {row['artists']}</div>
                                    <span class="music-badge">🏷️ {row['track_genre'].upper()}</span>
                                    <div class="sim-bar-container"><div class="sim-bar-fill" style="width: {sim_percentage}%;"></div></div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Hãy chọn bài hát để trải nghiệm.")

    with tab2:
        st.write("Biểu đồ đánh giá hiển thị ở đây (Đã cấu hình như phiên bản trước).")
