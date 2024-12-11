from searcher import SemanticSearcher

if __name__ == '__main__':
    searcher = SemanticSearcher()
    searcher.add_dataset(['why you play haruhikage', 'so what?', 'can i eat now'])
    print(searcher.search('eat?'))
