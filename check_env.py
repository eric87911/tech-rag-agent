import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_env():
    # 檢查點 1：.env 有沒有被讀到
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        logger.error("讀不到 GEMINI_API_KEY，請確認 .env 在專案根目錄，且終端機位置在專案根目錄")
        return False
    logger.info(f"API Key 已載入（開頭 {api_key[:6]}...），模型：{model_name}")

    # 檢查點 2：API 能不能打通
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="請用一句繁體中文回答：台積電的英文縮寫是什麼？",
        )
        logger.info(f"Gemini 回應成功：{response.text.strip()}")
        return True
    except Exception as e:
        logger.error(f"呼叫 Gemini 失敗：{type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    ok = check_env()
    print("✅ 環境建置完成" if ok else "❌ 環境有問題，請看上面的錯誤訊息")