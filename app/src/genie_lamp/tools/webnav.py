from selenium import webdriver
from selenium.webdriver.chrome.options import Options
def open_site(url: str):
    opts = Options(); opts.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    return {"ok": True}
