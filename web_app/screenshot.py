import asyncio
import argparse
from playwright.async_api import async_playwright

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:12345/")
    parser.add_argument("--out", default="screenshot.png")
    parser.add_argument("--wait", type=int, default=5000)
    parser.add_argument("--selector", default=None)
    parser.add_argument("--click", action="store_true")
    args = parser.parse_args()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Navigating to {args.url} ...")
        await page.goto(args.url)
        
        print(f"Waiting {args.wait}ms...")
        await asyncio.sleep(args.wait / 1000)
        
        if args.selector and args.click:
            print(f"Clicking {args.selector}...")
            await page.click(args.selector)
            await asyncio.sleep(1) # wait for modal/transition

        print(f"Taking screenshot to {args.out} ...")
        await page.screenshot(path=args.out)
        
        print("Done!")
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
