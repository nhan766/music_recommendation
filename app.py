import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. CHUẨN BỊ VÀ TIỀN XỬ LÝ DỮ LIỆU (Kaggle Dataset)
@st.cache_data
def load_data():
    # Đọc dữ liệu từ file CSV của Kaggle
    df = pd.read_csv("dataset.csv")
    
    # Giữ lại các cột quan trọng: Tên bài, Nghệ sĩ, Thể loại
    # (Lưu ý: Tên cột có thể thay đổi nhẹ tùy file CSV bạn tải, hãy mở file CSV kiểm tra nếu bị lỗi)
    df = df[['track_name', 'artists', 'track_genre']]
    
    # Xóa các dòng bị thiếu dữ liệu (NaN)
    df = df.dropna()
    
    # Xóa các bài hát bị trùng lặp tên để gợi ý đa dạng hơn
    df = df.drop_duplicates(subset=['track_name'])
    
    # MẸO QUAN TRỌNG: Lấy ngẫu nhiên 5000 bài hát để Demo. 
    # Tính toán Cosine cho 114.000 bài sẽ làm quá tải RAM của Streamlit miễn phí.
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    return df

df = load_data()

# 2. HÀM TÍNH TOÁN GỢI Ý (Content-based Filtering)
def recommend_songs(song_title, df, top_n=5):
    # Ghép Thể loại và Nghệ sĩ thành một chuỗi đặc trưng
    df['Features'] = df['track_genre'] + " " + df['artists']
    
    # Vector hóa đặc trưng bằng TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    
    # Tính ma trận độ tương đồng
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Tìm vị trí của bài hát người dùng nhập
    idx = df[df['track_name'] == song_title].index[0]
    
    # Lấy danh sách điểm tương đồng
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Lấy Top-N bài (bỏ bài đầu tiên vì là chính nó)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    
    return df.iloc[song_indices]

# 3. GIAO DIỆN NGƯỜI DÙNG
st.title("🎵 Music Recommendation System")
st.subheader("Team: NTH - IT Special Topic 1")

st.write(f"### 📂 Dữ liệu học máy: Đang chạy trên {len(df)} bài hát (Kaggle Dataset)")
with st.expander("Bấm để xem trước dữ liệu"):
    st.dataframe(df.head(10))

# Khung chọn bài hát
st.write("### 🎧 Chọn một bài hát làm cơ sở gợi ý:")
# Chỉ lấy danh sách tên bài hát để đưa vào Dropdown
song_list = df['track_name'].tolist()
selected_song = st.selectbox("Tìm kiếm bài hát:", song_list)

if st.button("Tạo danh sách gợi ý"):
    with st.spinner("Đang tính toán độ tương đồng ma trận..."):
        recommendations = recommend_songs(selected_song, df, top_n=5)
        
        st.success("Hoàn tất! Dưới đây là các bài hát tương đồng nhất:")
        # Đổi tên cột cho đẹp khi hiển thị
        recommendations = recommendations.rename(columns={
            'track_name': 'Tên bài hát', 
            'artists': 'Nghệ sĩ', 
            'track_genre': 'Thể loại'
        })
        st.table(recommendations[['Tên bài hát', 'Nghệ sĩ', 'Thể loại']])
