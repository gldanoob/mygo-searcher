import asyncio
from searcher import SemanticSearcher

async def main():
    searcher = SemanticSearcher()
    await searcher.add_sentences(['why you play haruhikage', 'so what?', 'can i eat now'])
    print(await searcher.search('eat?', k=1))

if __name__ == '__main__':
    asyncio.run(main())
