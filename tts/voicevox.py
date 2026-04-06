import asyncio
import aiohttp
import os
from typing import List, Dict, Any
from tts.base import BaseTTSEngine

class VoicevoxTTSEngine(BaseTTSEngine):
    """
    Async engine tạo audio từng dòng subtitle bằng Voicevox.
    """
    def __init__(
        self, 
        queue_tts: List[Dict[str, Any]], 
        voice_id: int = 10008,
        host: str = "127.0.0.1",
        port: int = 50121,
        # --- Tham số Voicevox mặc định ---
        speed_scale: float = 1.05,
        pitch_scale: float = -0.05,
        intonation_scale: float = 1.0,
        volume_scale: float = 2.0,
        pre_phoneme_length: float = None,  # Mặc định None
        post_phoneme_length: float = None, # Mặc định None
        output_sampling_rate: int = 48000,
        # --- Tham số hệ thống ---
        concurrent_requests: int = 20,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(queue_tts, **kwargs)
        self.voice_id = voice_id
        self.base_url = f"http://{host}:{port}"
        
        # Lưu trữ tham số Voicevox
        self.speed_scale = speed_scale
        self.pitch_scale = pitch_scale
        self.intonation_scale = intonation_scale
        self.volume_scale = volume_scale
        self.pre_phoneme_length = pre_phoneme_length
        self.post_phoneme_length = post_phoneme_length
        self.output_sampling_rate = output_sampling_rate
        
        # Lưu trữ tham số hệ thống
        self.concurrent_requests = concurrent_requests
        self.max_retries = max_retries

    async def _process_single(self, session: aiohttp.ClientSession, item: Dict[str, Any], index: int, semaphore: asyncio.Semaphore):
        """Xử lý 1 dòng subtitle với cơ chế Retry"""
        
        # Bỏ qua nếu text rỗng hoặc file đã tồn tại
        if not item.get('text', '').strip():
            return True
            
        wav_path = item['filename']
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return True

        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    # 1. Gọi /audio_query
                    async with session.post(
                        f"{self.base_url}/audio_query", 
                        params={"text": item['text'], "speaker": self.voice_id}
                    ) as query_res:
                        if query_res.status != 200:
                            raise Exception(f"Lỗi Query: {await query_res.text()}")
                        query_data = await query_res.json()
                    
                    # 2. Cập nhật tham số từ Class attributes
                    query_data['speedScale'] = self.speed_scale
                    query_data['pitchScale'] = self.pitch_scale
                    query_data['intonationScale'] = self.intonation_scale
                    query_data['volumeScale'] = self.volume_scale
                    query_data["outputSamplingRate"] = self.output_sampling_rate
                    
                    # Chỉ thêm pre/post PhonemeLength nếu chúng KHÔNG phải là None
                    if self.pre_phoneme_length is not None:
                        query_data["prePhonemeLength"] = self.pre_phoneme_length
                    if self.post_phoneme_length is not None:
                        query_data["postPhonemeLength"] = self.post_phoneme_length
                    
                    # 3. Gọi /synthesis
                    async with session.post(
                        f"{self.base_url}/synthesis", 
                        params={"speaker": self.voice_id}, 
                        json=query_data
                    ) as synth_res:
                        if synth_res.status != 200:
                            raise Exception(f"Lỗi Synthesis: {await synth_res.text()}")
                        audio_content = await synth_res.read()
                        
                    # 4. Ghi file thành công -> Thoát vòng lặp retry
                    with open(wav_path, "wb") as f:
                        f.write(audio_content)
                    
                    return True # Thành công

                except Exception as e:
                    print(f"[{index}] ⚠️ Lỗi (Lần thử {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1) # Nghỉ 1 giây trước khi thử lại
                    else:
                        print(f"❌ Thất bại hoàn toàn câu {index} sau {self.max_retries} lần thử.")
                        return False # Thất bại

    async def _run_async(self):
        """Chạy toàn bộ queue song song"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [
                self._process_single(session, item, i+1, semaphore) 
                for i, item in enumerate(self.queue_tts)
            ]
            results = await asyncio.gather(*tasks)
            
            # Thống kê kết quả
            ok_count = sum(1 for r in results if r is True)
            err_count = len(results) - ok_count
            return {"ok": ok_count, "err": err_count}

    def run(self) -> Dict[str, int]:
        """Entry point đồng bộ"""
        print(f"[VoicevoxTTS] Bắt đầu {len(self.queue_tts)} dòng | voice_id={self.voice_id}")
        return asyncio.run(self._run_async())
