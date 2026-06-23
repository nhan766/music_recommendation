import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import hashlib
from datetime import datetime

# 1. CẤU HÌNH TRANG WEB & CSS 

st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .music-card { 
        background: linear-gradient(145deg, #1e222b, #15181f); 
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.2); 
        border: 1px solid #2d3139; 
    }
    .music-title { color: #ffffff; font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    .music-artist { color: #b3b3b3; font-size: 14px; margin-bottom: 10px; }
    .music-badge { background-color: #282c34; color: #1db954; padding: 4px 10px; border-radius: 20px; font-size: 11px; }
    .sim-bar-container { background-color: #282c34; border-radius: 10px; height: 6px; width: 100%; margin-top: 10px; }
    .sim-bar-fill { background: linear-gradient(90deg, #1db954, #1ed760); height: 6px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# 2. HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU (SQLite)

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable (username TEXT, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS historytable (username TEXT, song_title TEXT, search_time TEXT)')
    conn.commit()
    return conn

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
    # Kiểm tra tránh trùng lặp liên tiếp một bài hát trong lịch sử ngắn hạn
    c.execute('SELECT song_title FROM historytable WHERE username = ? ORDER BY search_time DESC LIMIT 1', (username,))
    last_song = c.fetchone()
    if not last_song or last_song[0] != song_title:
        c.execute('INSERT INTO historytable (username, song_title, search_time) VALUES (?,?,?)', (username, song_title, now))
        conn.commit()

def get_user_history(username):
    conn = init_db()
    c = conn.cursor()
    c.execute('SELECT song_title, search_time FROM historytable WHERE username = ? ORDER BY search_time DESC LIMIT 10', (username,))
    return c.fetchall()

init_db()

# 3. THUẬT TOÁN LOGIC HỆ THỐNG GỢI Ý (Hybrid Model)

@st.cache_data
def load_data():
    try:
        # Sử dụng tệp dữ liệu Spotify thu thập từ nguồn Kaggle
        df = pd.read_csv("dataset.csv")
        if 'popularity' not in df.columns:
            np.random.seed(42)
            df['popularity'] = np.random.randint(10, 100, size=len(df))
        df = df[['track_name', 'artists', 'track_genre', 'popularity']].dropna()
        df = df.drop_duplicates(subset=['track_name'])
        # Trích xuất tập mẫu ngẫu nhiên 5000 bài hát để tối ưu hiệu năng tính toán
        return df.sample(n=5000, random_state=42).reset_index(drop=True)
    except:
        # Dữ liệu dự phòng cục bộ
        data = {
            'track_name': ['Shape of You', 'Bohemian Rhapsody', 'Perfect', "It's Your Birthday"],
            'artists': ['Ed Sheeran', 'Queen', 'Ed Sheeran', 'Allman Brown'],
            'track_genre': ['Pop', 'Rock', 'Pop', 'Latino'],
            'popularity': [95, 90, 88, 75]
        }
        return pd.DataFrame(data)

df = load_data()

def hybrid_recommend(song_title, df, top_n=4, content_weight=0.7, pop_weight=0.3):
    # Trích xuất đặc trưng văn bản bằng TF-IDF
    df['Features'] = df['track_genre'] + " " + df['artists']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Chuẩn hóa thuộc tính độ phổ biến về thang điểm [0, 1] để làm Lọc cộng tác proxy
    normalized_pop = df['popularity'] / 100.0
    idx = df[df['track_name'] == song_title].index[0]
    
    # Tính điểm lai tuyến tính linh hoạt: s(u,i) = (α * Content) + ((1-α) * Popularity)
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


# 4. QUẢN LÝ PHIÊN (Session State) & ĐIỀU HƯỚNG UI

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'search_song_input' not in st.session_state:
    st.session_state['search_song_input'] = None
if 'trigger_search' not in st.session_state:
    st.session_state['trigger_search'] = False

# --- TRANG XÁC THỰC: ĐĂNG NHẬP / ĐĂNG KÝ (Center Card Layout) ---
if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #1db954; font-size: 3em;'>🎵 Music Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b3b3b3; margin-bottom: 40px;'>Đăng nhập để khám phá và lưu trữ không gian âm nhạc của riêng bạn.</p>", unsafe_allow_html=True)
    
    col1, col_center, col3 = st.columns([1, 1.2, 1])
    with col_center:
        tab_login, tab_register = st.tabs(["🔐 Đăng nhập", "📝 Đăng ký tài khoản mới"])
        
        with tab_login:
            st.markdown("### Mừng bạn trở lại!")
            username = st.text_input("Tên đăng nhập", placeholder="Nhập username của bạn...", key="login_user")
            password = st.text_input("Mật khẩu", placeholder="Nhập mật khẩu...", type='password', key="login_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Đăng nhập ngay", type="primary", use_container_width=True):
                if username and password:
                    hashed_pswd = make_hashes(password)
                    result = login_user(username, hashed_pswd)
                    if result:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun() 
                    else:
                        st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng!")
                else:
                    st.warning("Vui lòng điền đầy đủ thông tin.")

        with tab_register:
            st.markdown("### Trở thành thành viên")
            new_user = st.text_input("Tên đăng nhập mới", placeholder="Chọn một username độc đáo...", key="reg_user")
            new_password = st.text_input("Mật khẩu mới", placeholder="Tạo mật khẩu an toàn...", type='password', key="reg_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Tạo tài khoản", use_container_width=True):
                if new_user and new_password:
                    conn = init_db()
                    c = conn.cursor()
                    c.execute('SELECT * FROM userstable WHERE username = ?', (new_user,))
                    if c.fetchone():
                        st.error("⚠️ Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")
                    else:
                        add_user(new_user, make_hashes(new_password))
                        st.success("✅ Tạo tài khoản thành công! Hãy chuyển sang thẻ Đăng nhập.")
                else:
                    st.warning("Vui lòng điền đầy đủ thông tin.")

# --- TRANG CHỦ ---
else:
    # Hàm xử lý bắt sự kiện khi ấn vào mục lịch sử bên Sidebar
    def click_history_callback(song_name):
        st.session_state['search_song_input'] = song_name
        st.session_state['trigger_search'] = True

    # Thanh Sidebar bên trái quản lý hồ sơ và Lịch sử tương tác
    with st.sidebar:
        st.markdown(f"### 👤 Xin chào, **<span style='color:#1db954;'>{st.session_state['username']}</span>**", unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['search_song_input'] = None
            st.session_state['trigger_search'] = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🕒 Lịch sử khám phá (Ấn để xem lại)")
        history_data = get_user_history(st.session_state['username'])
        
        if history_data:
            for song, time_str in history_data:
                btn_label = f"🎵 {song}\n⏱️ {time_str}"
                
                # Tạo khóa định danh tĩnh, sạch để gắn CSS
                button_key = f"hist_{time_str}_{song}".replace(" ", "_").replace(":", "_").replace("-", "_")
                
                # Ép kiểu giao diện CSS động cho từng nút lịch sử riêng biệt
                st.html(f"""
                    <style>
                    div[data-testid="stSidebar"] button[key="{button_key}"] {{
                        background-color: #1e222b !important;
                        color: #ffffff !important;
                        border: none !important;
                        border-left: 3px solid #1db954 !important;
                        border-radius: 8px !important;
                        padding: 10px !important;
                        text-align: left !important;
                        width: 100% !important;
                        display: block !important;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                    }}
                    div[data-testid="stSidebar"] button[key="{button_key}"]:hover {{
                        background-color: #252a36 !important;
                        border-left: 3px solid #1ed760 !important;
                    }}
                    </style>
                """)
                
                # Khởi tạo nút bấm lịch sử chuẩn cấu trúc Streamlit API
                st.button(
                    btn_label, 
                    key=button_key, 
                    help=f"Xem lại gợi ý cho bài {song}",
                    on_click=click_history_callback,
                    args=(song,)
                )
        else:
            st.info("Bạn chưa tìm kiếm bài hát nào.")

    # Không gian hiển thị chức năng chính (Bố cục 2 cột)
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>🎵 Hybrid Music Recommendation System</h1>", unsafe_allow_html=True)
    
    col_input, col_spacer, col_output = st.columns([1.5, 0.2, 3])
    
    with col_input:
        st.markdown("<h3 style='color: #ffffff;'>⚙️ Trình điều khiển</h3>", unsafe_allow_html=True)
        song_list = df['track_name'].tolist()
        
        # Định vị trí đồng bộ ô Tìm kiếm tự động theo Lịch sử Sidebar
        default_index = None
        if st.session_state['search_song_input'] in song_list:
            default_index = song_list.index(st.session_state['search_song_input'])
            
        selected_song = st.selectbox(
            "🔍 Nhập tên bài hát:", 
            options=song_list, 
            index=default_index, 
            placeholder="Ví dụ: Shape of You"
        )
        
        # Nếu người dùng thay đổi lựa chọn thủ công trên Dropdown, reset cờ tự động kích hoạt
        if selected_song != st.session_state['search_song_input']:
            st.session_state['search_song_input'] = selected_song
            st.session_state['trigger_search'] = False
        
        st.markdown("<br><p style='color: #b3b3b3; font-size: 14px;'>Điều chỉnh Trọng số Hybrid:</p>", unsafe_allow_html=True)
        content_weight = st.slider("Tỷ trọng Lọc nội dung (Content-based)", 0.0, 1.0, 0.7)
        pop_weight = 1.0 - content_weight
        st.caption(f"Tỷ trọng Lọc cộng tác/Phổ biến: **{pop_weight:.1f}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        search_pressed = st.button("Phân tích & Gợi ý", type="primary", use_container_width=True, disabled=(selected_song is None))
        if search_pressed:
            st.session_state['trigger_search'] = True

    with col_output:
        if st.session_state['trigger_search'] and st.session_state['search_song_input']:
            current_search_song = st.session_state['search_song_input']
            
            # Ghi vết tiến trình vào database SQLite
            add_search_history(st.session_state['username'], current_search_song)
            
            st.markdown(f"<h3 style='color: #ffffff;'>✨ Gợi ý Top-N cho: {current_search_song}</h3>", unsafe_allow_html=True)
            with st.spinner("Đang tính toán ma trận tương tác $R$ và tối ưu Hybrid Score..."):
                recommendations = hybrid_recommend(current_search_song, df, top_n=4, content_weight=content_weight, pop_weight=pop_weight)
                
                # Phân phối kết quả đầu ra theo kiến trúc lưới (2 dòng x 2 cột) giống Spotify thật
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
                                <div style="color: #888; font-size: 11px; margin-top: 12px; display: flex; justify-content: space-between;">
                                    <span>Điểm Hybrid ($s(u,i)$)</span>
                                    <span style="color: #1db954; font-weight: bold;">{sim_percentage}%</span>
                                </div>
                                <div class="sim-bar-container"><div class="sim-bar-fill" style="width: {sim_percentage}%;"></div></div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Hệ thống giải quyết bài toán Cold-Start bằng cách kết hợp đặc trưng item và độ phổ biến. Hãy gõ tên bài hát hoặc chọn một mục từ Lịch sử khám phá bên trái.")
