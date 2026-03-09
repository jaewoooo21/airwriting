import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Capture console messages
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

        print("Navigating to http://localhost:12345/ ...")
        await page.goto('http://localhost:12345/')
        
        print("Waiting 3 seconds...")
        await asyncio.sleep(3)
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
