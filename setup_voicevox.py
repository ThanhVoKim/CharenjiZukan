import os
import glob
import shutil
import subprocess
import time
import sys

def main():
    print("=======================================================")
    print("  🚀 BẮT ĐẦU QUY TRÌNH SETUP VOICEVOX NEMO ENGINE 🚀  ")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # PHẦN 1: CHUẨN BỊ MÃ NGUỒN VÀ DỮ LIỆU
    # ---------------------------------------------------------
    print("🚀 1. Tải mã nguồn và cài đặt UV...")
    if not os.path.exists("/content/voicevox_nemo_engine"):
        os.system("git clone -q https://github.com/VOICEVOX/voicevox_nemo_engine.git /content/voicevox_nemo_engine")

    os.chdir("/content/voicevox_nemo_engine")
    os.system("curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL='/usr/local/bin' sh")
    os.system("uv sync --quiet")

    print("🚀 2. Tải Core AI (GPU)...")
    os.system("apt-get install jq -y -q > /dev/null 2>&1")
    os.system("curl -s https://api.github.com/repos/VOICEVOX/voicevox_nemo_core/releases/latest | jq -r '.assets[] | select(.name | contains(\"linux\") and contains(\"gpu\") and contains(\"zip\")) | .browser_download_url' | xargs wget -qO core.zip")
    os.system("unzip -qo core.zip -d ./nemo_core")

    print("🚀 3. Tải dữ liệu nhân vật...")
    os.system("curl -s https://api.github.com/repos/VOICEVOX/voicevox_nemo_engine/releases/latest | jq -r '.assets[] | select(.name | contains(\"linux\") and contains(\"cpu\") and contains(\".vvpp\") and (contains(\".txt\") | not)) | .browser_download_url' | head -n 1 | xargs wget -qO engine_release.vvpp")
    os.system("unzip -qo engine_release.vvpp -d ./temp_fix")

    # Trích xuất thư mục nhân vật
    char_dirs = glob.glob("./temp_fix/**/resources/character_info", recursive=True)
    if char_dirs:
        if os.path.exists("./resources/character_info"):
            shutil.rmtree("./resources/character_info")
        shutil.copytree(char_dirs[0], "./resources/character_info")
        print("✅ Đã nạp thành công dữ liệu nhân vật!")

    os.system("rm -rf ./temp_fix engine_release.vvpp core.zip")
    print("🎉 Hoàn tất bước chuẩn bị dữ liệu!\n")

    # ---------------------------------------------------------
    # PHẦN 2: XỬ LÝ CUDA VÀ MÔI TRƯỜNG LÕI
    # ---------------------------------------------------------
    print("🧹 Dọn dẹp tiến trình cũ...")
    os.system("pkill -f 'run.py'")
    time.sleep(2)

    print("📦 4. Tải thư viện CUDA 11 & ép tải cuDNN bản 8...")
    install_cmd = 'pip install "nvidia-cudnn-cu11<9.0" nvidia-cufft-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-cusparse-cu11 nvidia-curand-cu11'
    os.system(install_cmd)

    print("\n🔍 5. KIỂM TRA ĐỘ CHÍNH XÁC CỦA THƯ VIỆN cuDNN...")
    result = subprocess.run(["pip", "show", "nvidia-cudnn-cu11"], capture_output=True, text=True)

    if "WARNING" in result.stderr or "Name: nvidia-cudnn-cu11" not in result.stdout:
        print("❌ LỖI: Không tìm thấy gói nvidia-cudnn-cu11. Quá trình tải pip đã thất bại!")
        sys.exit(1)

    # Trích xuất Location một cách an toàn
    location = ""
    for line in result.stdout.split('\n'):
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
            break

    if not location:
         print("❌ LỖI: Không tìm được đường dẫn Location.")
         sys.exit(1)

    print(f"   -> Thư mục cài đặt: {location}")

    check_cmd = f"find {location} -name 'libcudnn.so.8*'"
    found_files = subprocess.getoutput(check_cmd).strip()

    if not found_files or "No such file" in found_files:
        print("❌ LỖI CHÍ MẠNG: Thư mục tồn tại nhưng KHÔNG CÓ file libcudnn.so.8!")
        sys.exit(1)
    else:
        print(f"✅ BƯỚC KIỂM TRA HOÀN HẢO! Đã tìm thấy đúng cuDNN bản 8 tại:\n{found_files}\n")

    print("🔍 6. COPY VẬT LÝ THƯ VIỆN VÀO LÕI HỆ ĐIỀU HÀNH...")
    os.system(f"cp -L {location}/nvidia/*/lib/*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null")
    os.system("ldconfig 2>/dev/null")

    # Kiểm tra xác nhận ở lõi Linux
    check_file = subprocess.getoutput("ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so.8")
    if "No such file" in check_file:
        print("❌ LỖI: Vẫn không thấy cuDNN 8 sau khi copy.")
        sys.exit(1)
    else:
        print(f"✅ ĐÃ ÉP CÀI ĐẶT VÀO LÕI THÀNH CÔNG:\n{check_file}")

if __name__ == "__main__":
    main()