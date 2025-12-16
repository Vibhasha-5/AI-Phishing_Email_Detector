import re
import numpy as np
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "verify", "urgent", "click", "login", "password",
    "bank", "account", "confirm", "security", "update"
]

def extract_url_features(text):
    urls = re.findall(r"https?://[^\s]+", str(text))
    num_urls = len(urls)

    has_ip_url = 0
    https_count = 0

    for url in urls:
        parsed = urlparse(url)
        if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc):
            has_ip_url = 1
        if url.startswith("https"):
            https_count += 1

    suspicious_word_count = sum(
        1 for word in SUSPICIOUS_WORDS if word in str(text).lower()
    )

    return np.array([
        num_urls,
        has_ip_url,
        https_count,
        suspicious_word_count
    ])

def url_feature_transformer(texts):
    return np.vstack([extract_url_features(text) for text in texts])
