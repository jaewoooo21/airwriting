import asyncio
import time
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to http://localhost:12345/ ...")
        await page.goto('http://localhost:12345/')
        
        print("Waiting 5 seconds for intro to finish...")
        await asyncio.sleep(5)
        
        print("Taking screenshot...")
        await page.screenshot(path='c:/Users/USER/airwriting_imu_only/platform_app/debug_screenshot.png')
        
        print("Done!")
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
