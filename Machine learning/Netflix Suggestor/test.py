import pandas as pd

titles_db = pd.read_csv('title.csv')
for title  in titles_db.title:
    print(f'"{title}",')