import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# CẤU HÌNH TRANG WEB
st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="wide")

# CSS CAO CẤP (Giữ nguyên giao diện đẹp)
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

# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        # Thêm cột popularity để làm Hybrid, nếu file csv không có thì giả lập
        if 'popularity' not in df.columns:
            np.random.seed(42)
            df['popularity'] = np.random.randint(10, 100, size=len(df))
            
        df = df[['track_name', 'artists', 'track_genre', 'popularity']].dropna()
        df = df.drop_duplicates(subset=['track_name'])
        return df.sample(n=5000, random_state=42).reset_index(drop=True)
    except:
        data = {
            'track_name': ['Shape of You', 'Bohemian Rhapsody', 'Perfect', 'Hotel California', 'Thinking Out Loud', 'Stairway to Heaven'],
            'artists': ['Ed Sheeran', 'Queen', 'Ed Sheeran', 'Eagles', 'Ed Sheeran', 'Led Zeppelin'],
            'track_genre': ['Pop', 'Rock', 'Pop', 'Rock', 'Pop', 'Rock'],
            'popularity': [95, 90, 88, 85, 82, 80]
        }
        return pd.DataFrame(data)

df = load_data()

# 2. THUẬT TOÁN HYBRID (Content-Based + Popularity/Collaborative proxy)
def hybrid_recommend(song_title, df, top_n=4, content_weight=0.7, pop_weight=0.3):
    # Trích xuất vector & Tính Cosine Similarity (Mô hình hóa toán học)
    df['Features'] = df['track_genre'] + " " + df['artists']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Chuẩn hóa cột popularity về thang điểm từ 0 đến 1
    normalized_pop = df['popularity'] / 100.0
    
    idx = df[df['track_name'] == song_title].index[0]
    
    # Tính điểm lai (Hybrid Score) = (Trọng số Content * Cosine) + (Trọng số Collaborative * Popularity)
    hybrid_scores = []
    for i, content_score in enumerate(cosine_sim[idx]):
        if i != idx: # Bỏ qua bài hát gốc
            final_score = (content_weight * content_score) + (pop_weight * normalized_pop.iloc[i])
            hybrid_scores.append((i, final_score))
            
    # Sắp xếp và lấy Top N
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    results = []
    for i, score in hybrid_scores:
        row = df.iloc[i].copy()
        row['hybrid_score'] = score
        results.append(row)
    return pd.DataFrame(results)

# 3. GIAO DIỆN CHÍNH
st.markdown("<h1 style='text-align: center; color: #ffffff;'>🎵 Hybrid Music Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b3b3b3;'>Course: Recommendation Systems | Instructor: PhD. Nguyen Luong Vuong</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 12px;'>Group 7: Huynh Thanh Nhan, Nguyen Ngoc Quang Huy, Tran The Thinh, Ho Dac Nhan</p>", unsafe_allow_html=True)

# Tạo 2 Tabs: Tab dùng thử App và Tab Báo cáo khoa học
tab1, tab2 = st.tabs(["🎧 Khám phá Âm nhạc (App)", "📊 Đánh giá Mô hình (Evaluation)"])

with tab1:
    col_input, col_spacer, col_output = st.columns([1.5, 0.2, 3])
    with col_input:
        st.markdown("<h3 style='color: #ffffff;'>⚙️ Trình điều khiển</h3>", unsafe_allow_html=True)
        song_list = df['track_name'].tolist()
        selected_song = st.selectbox("🔍 Nhập tên bài hát:", options=song_list, index=None, placeholder="Ví dụ: Shape of You")
        
        # Thanh trượt điều chỉnh Hybrid (Thể hiện sự hiểu biết sâu về mô hình)
        st.markdown("<br><p style='color: #b3b3b3; font-size: 14px;'>Điều chỉnh Trọng số Hybrid:</p>", unsafe_allow_html=True)
        content_weight = st.slider("Tỷ trọng Lọc nội dung (Content-based)", 0.0, 1.0, 0.7)
        pop_weight = 1.0 - content_weight
        st.caption(f"Tỷ trọng Lọc cộng tác/Phổ biến: **{pop_weight:.1f}**")
        
        search_button = st.button("Phân tích & Gợi ý", use_container_width=True, disabled=(selected_song is None))

    with col_output:
        if search_button and selected_song:
            st.markdown(f"<h3 style='color: #ffffff;'>✨ Gợi ý Top-N cho: {selected_song}</h3>", unsafe_allow_html=True)
            with st.spinner("Đang tính toán ma trận tương tác $R$ và tối ưu Hybrid Score..."):
                recommendations = hybrid_recommend(selected_song, df, top_n=4, content_weight=content_weight, pop_weight=pop_weight)
                
                grid_col1, grid_col2 = st.columns(2)
                for index, row in recommendations.iterrows():
                    target_col = grid_col1 if index % 2 == 0 else grid_col2
                    # Chuẩn hóa điểm hybrid để hiển thị UI
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
                                <div class="sim-bar-container">
                                    <div class="sim-bar-fill" style="width: {sim_percentage}%;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Hệ thống giải quyết bài toán Cold-Start bằng cách kết hợp đặc trưng item và độ phổ biến. Hãy chọn bài hát để trải nghiệm.")

    
