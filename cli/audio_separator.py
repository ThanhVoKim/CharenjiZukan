"""
audio_separator.py — CLI: Tách voice/background từ audio sử dụng thư viện audio-separator
"""
import os
import sys
import yaml
import logging
import argparse
import shutil
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path="config/audio_separator_config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file config {config_path}: {e}")
        return {}

def separate_audio(input_path, output_dir, preset="bgm_extraction", config_path="config/audio_separator_config.yaml", override_kwargs=None):
    """
    Tách audio thành Instrumental hoặc Vocals sử dụng audio-separator.
    
    Args:
        input_path: Đường dẫn file audio gốc.
        output_dir: Thư mục lưu kết quả.
        preset: 'bgm_extraction' (lấy Instrumental) hoặc 'vocal_extraction' (lấy Vocals)
        config_path: Đường dẫn tới file yaml cấu hình.
        override_kwargs: Dictionary chứa các tham số ghi đè (ví dụ từ render_config.json)
        
    Returns:
        str: Đường dẫn file output đã chọn (Instrumental hoặc Vocals)
    """
    from audio_separator.separator import Separator
    
    config = load_config(config_path)
    if preset not in config:
        raise ValueError(f"Preset {preset} không tồn tại trong file cấu hình.")
        
    # Tạo bản sao để tránh sửa trực tiếp config đang load
    preset_config = config[preset].copy()
    model_name = preset_config.pop("model")
    
    logger.info(f"Khởi tạo Separator cho preset '{preset}' với mô hình: {model_name}")
    
    separator_args = {
        "output_dir": output_dir,
        "output_format": "WAV",
    }
    
    # Merge thêm kwargs
    separator_args.update(preset_config)
    
    if override_kwargs:
        # Nếu cùng có mdxc_params thì merge sâu
        if "mdxc_params" in override_kwargs and "mdxc_params" in separator_args:
            separator_args["mdxc_params"].update(override_kwargs["mdxc_params"])
            override_copy = override_kwargs.copy()
            override_copy.pop("mdxc_params")
            separator_args.update(override_copy)
        else:
            separator_args.update(override_kwargs)
            
        # Xóa những khóa không liên quan đến cấu hình Separator
        separator_args.pop("extract_bgm", None)
        separator_args.pop("extract_vocals", None)
    
    separator = Separator(**separator_args)
    separator.load_model(model_name)
    
    logger.info(f"Đang xử lý file: {input_path}")
    output_files = separator.separate(input_path)
    
    target_keyword = "Instrumental" if preset == "bgm_extraction" else "Vocals"
    target_file = None
    
    for f in output_files:
        abs_f = os.path.join(output_dir, f)
        if target_keyword in f:
            target_file = abs_f
        else:
            # Xóa các file không cần thiết (stem thừa)
            try:
                os.remove(abs_f)
            except Exception:
                pass
                
    if not target_file:
        logger.warning(f"Không tìm thấy file có chứa chữ '{target_keyword}'. Output files: {output_files}")
        if output_files:
            target_file = os.path.join(output_dir, output_files[0])
            
    logger.info(f"Tách audio hoàn tất. Kết quả: {target_file}")
    return target_file

def separate_audio_batch(input_paths, output_paths, preset="vocal_extraction", config_path="config/audio_separator_config.yaml", override_kwargs=None):
    """
    Xử lý theo batch nhiều file âm thanh để tiết kiệm thời gian load mô hình.
    """
    from audio_separator.separator import Separator
    
    config = load_config(config_path)
    if preset not in config:
        raise ValueError(f"Preset {preset} không tồn tại trong file cấu hình.")
        
    preset_config = config[preset].copy()
    model_name = preset_config.pop("model")
    
    if not output_paths:
        return []
        
    base_output_dir = os.path.dirname(output_paths[0])
    
    separator_args = {
        "output_dir": base_output_dir,
        "output_format": "WAV",
    }
    separator_args.update(preset_config)
    
    if override_kwargs:
        if "mdxc_params" in override_kwargs and "mdxc_params" in separator_args:
            separator_args["mdxc_params"].update(override_kwargs["mdxc_params"])
            override_copy = override_kwargs.copy()
            override_copy.pop("mdxc_params")
            separator_args.update(override_copy)
        else:
            separator_args.update(override_kwargs)
            
        separator_args.pop("extract_bgm", None)
        separator_args.pop("extract_vocals", None)
    
    logger.info(f"Khởi tạo Separator cho BATCH processing, preset '{preset}', mô hình: {model_name}")
    separator = Separator(**separator_args)
    separator.load_model(model_name)
    
    results = []
    target_keyword = "Instrumental" if preset == "bgm_extraction" else "Vocals"
    
    for in_path, out_path in zip(input_paths, output_paths):
        logger.debug(f"Đang xử lý: {in_path}")
        curr_out_dir = os.path.dirname(out_path)
        separator.output_dir = curr_out_dir
        
        output_files = separator.separate(in_path)
        
        found_target = None
        for f in output_files:
            abs_f = os.path.join(curr_out_dir, f)
            if target_keyword in f:
                shutil.move(abs_f, out_path)
                found_target = out_path
            else:
                try:
                    os.remove(abs_f)
                except Exception:
                    pass
                    
        if found_target:
            results.append(found_target)
        else:
            logger.warning(f"Không tìm thấy {target_keyword} cho input {in_path}")
            if output_files:
                backup = os.path.join(curr_out_dir, output_files[0])
                shutil.move(backup, out_path)
                results.append(out_path)
                for f in output_files[1:]:
                    try:
                        os.remove(os.path.join(curr_out_dir, f))
                    except: pass
            
    logger.info(f"Hoàn tất batch processing cho {len(results)} files.")
    return results

def main():
    parser = argparse.ArgumentParser(description="Tách âm thanh sử dụng audio-separator")
    parser.add_argument("--input", "-i", required=True, help="File input")
    parser.add_argument("--output-dir", "-o", default=".", help="Thư mục output")
    parser.add_argument("--preset", "-p", choices=["bgm_extraction", "vocal_extraction"], default="bgm_extraction")
    args = parser.parse_args()
    
    separate_audio(args.input, args.output_dir, preset=args.preset)

if __name__ == "__main__":
    main()