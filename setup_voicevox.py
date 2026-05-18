import os
import glob
import shutil
import subprocess
import time
import sys

def main():
    print("=======================================================")
    print("  🚀 BẮT ĐẦU QUY TRÌNH SETUP VOICEVOX ENGINE CHUẨN 🚀  ")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # PHẦN 1: CHUẨN BỊ MÃ NGUỒN VÀ MÔI TRƯỜNG
    # ---------------------------------------------------------
    print("🚀 1. Tải mã nguồn và cài đặt UV...")
    if not os.path.exists("/content/voicevox_engine"):
        os.system("git clone -q https://github.com/VOICEVOX/voicevox_engine.git /content/voicevox_engine")

    os.chdir("/content/voicevox_engine")
    os.system("curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL='/usr/local/bin' sh")
    os.system("uv sync --quiet")

    # ---------------------------------------------------------
    # PHẦN 2: TẢI CORE AI VÀ TỪ ĐIỂN OPEN JTALK
    # ---------------------------------------------------------
    print("🚀 2. Tải Core AI (GPU) và xác nhận Điều khoản...")
    os.system("rm -rf core core.zip download-linux-x64")
    os.system("wget -q https://github.com/VOICEVOX/voicevox_core/releases/latest/download/download-linux-x64")
    os.system("chmod +x download-linux-x64")

    # TỰ ĐỘNG ĐỒNG Ý ĐIỀU KHOẢN SỬ DỤNG
    os.system('yes "y" | ./download-linux-x64 --devices cuda --output ./core > /dev/null 2>&1')

    print("🚀 3. Tải OpenJTalk Dictionary...")
    if not os.path.exists("open_jtalk_dic"):
        os.system("wget -qO open_jtalk_dic_utf_8-1.11.tar.gz https://sourceforge.net/projects/open-jtalk/files/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz/download")
        os.system("tar xzf open_jtalk_dic_utf_8-1.11.tar.gz")
        os.system("mv open_jtalk_dic_utf_8-1.11 open_jtalk_dic")
        os.system("rm -f open_jtalk_dic_utf_8-1.11.tar.gz")

    # ---------------------------------------------------------
    # PHẦN 3: XỬ LÝ CUDA VÀ MÔI TRƯỜNG LÕI
    # ---------------------------------------------------------
    print("📦 4. Ép tải thư viện cuDNN bản 8 vào lõi Linux...")
    os.system('pip install -q "nvidia-cudnn-cu11<9.0" nvidia-cufft-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-cusparse-cu11 nvidia-curand-cu11')
    result = subprocess.run(["pip", "show", "nvidia-cudnn-cu11"], capture_output=True, text=True)

    location = ""
    for line in result.stdout.split('\n'):
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
            break

    if location:
        os.system(f"cp -L {location}/nvidia/*/lib/*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null")
        os.system("ldconfig 2>/dev/null")

    # ---------------------------------------------------------
    # PHẦN 4: QUY HOẠCH LẠI CẤU TRÚC LÕI VÀ VÁ LỖI MÃ NGUỒN
    # ---------------------------------------------------------
    print("🧹 5. Sắp xếp lại cấu trúc Lõi và Model...")
    final_core_dir = "/content/voicevox_engine/final_core"
    final_model_dir = os.path.join(final_core_dir, "model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Gom .so vào final_core
    for f in glob.glob("/content/voicevox_engine/core/**/*.so*", recursive=True):
        try: shutil.copy(f, final_core_dir)
        except: pass

    # Gom .vvm vào final_core/model
    for f in glob.glob("/content/voicevox_engine/core/**/*.vvm", recursive=True):
        try: shutil.copy(f, final_model_dir)
        except: pass

    # Copy các file thư viện .so vào hệ thống hệ điều hành (fix lỗi ONNX Runtime)
    os.system(f"cp -d {final_core_dir}/*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null")
    os.system("ldconfig 2>/dev/null")

    print("🛠️ 6. Vá lỗi (Patch) mã nguồn bỏ qua các nhân vật thiếu Metadata...")
    metas_store_path = "/content/voicevox_engine/voicevox_engine/metas/metas_store.py"
    if os.path.exists(metas_store_path):
        with open(metas_store_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "engine_character = self._loaded_metas[character_uuid]" in line:
                indent = line[:line.find("engine_character")]
                # Sửa thành .get() và lơ đi nếu không có
                lines[i] = f"{indent}engine_character = self._loaded_metas.get(character_uuid)\n{indent}if engine_character is None:\n{indent}    continue\n"

        with open(metas_store_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    # ---------------------------------------------------------
    # PHẦN 5: KHỞI ĐỘNG SERVER NGẦM
    # ---------------------------------------------------------
    print("\n🚀 7. Khởi động Server ngầm...")
    os.system("pkill -f 'run.py'")
    time.sleep(2)

    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = f"{final_core_dir}:{my_env.get('LD_LIBRARY_PATH', '')}"

    log_file = open("engine.log", "w")

    # CẮT ĐỨT 100% LIÊN KẾT VỚI COLAB ĐỂ CHẠY NGẦM
    process = subprocess.Popen(
        ["uv", "run", "run.py",
         "--use_gpu",
         "--host", "127.0.0.1",
         "--port", "50021",
         "--voicelib_dir", final_core_dir],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,  # Chặn luồng gõ phím từ Colab
        start_new_session=True,    # Đẩy Server thành một Group độc lập
        close_fds=True,            # Đóng toàn bộ các cổng kết nối thừa
        env=my_env
    )

    log_file.close()

    print("⏳ Đang đợi Server khởi động...")
    with open("engine.log", "r") as f:
        while True:
            line = f.readline()
            if line:
                # print(line.strip()) # Ẩn log để màn hình sạch sẽ
                if "Uvicorn running on" in line or "Application startup complete" in line:
                    print("-" * 50)
                    print("\n🎉 THÀNH CÔNG TUYỆT ĐỐI! SERVER ĐANG CHẠY NGẦM (GPU CUDA).")
                    print("👉 File Setup kết thúc tại đây. Colab sẽ nhả cell ngay bây giờ!")
                    break
            else:
                if process.poll() is not None:
                    print("\n❌ Server bị Crash. Hãy xem file engine.log.")
                    os.system("tail -n 15 engine.log")
                    break
                time.sleep(0.5)

    # Ép Python Script kết thúc dứt điểm ngay tại đây, nhả Cell cho Colab
    sys.exit(0)

if __name__ == "__main__":
    main()
