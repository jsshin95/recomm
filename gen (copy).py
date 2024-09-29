#import pandas as pd
import random

#df = pd.read_csv('img/1_books/1_books_items.csv',delimiter='\t')
#df = pd.read_csv('img/2_electronics/2_electronics_items.csv',delimiter='\t')
#df = pd.read_csv('img/3_clothing/3_clothing_items.csv',delimiter='\t')
#df = pd.read_csv('img/4_food/4_food_items.csv',delimiter='\t')
#df = pd.read_csv('img/5_life/5_life_items.csv',delimiter='\t')
#df = pd.read_csv('img/6_beauty/6_beauty_items.csv',delimiter='\t')
#print(df)

"""
for i in range(50):
    print('INSERT INTO item (org_price, dis_price, name, thumbnail_url, is_recommended, is_soldout, seller_id, category_id)')
    print('VALUES (%d, %d, \'%s\', "https://commerce-side.s3.ap-northeast-2.amazonaws.com/items/thumbnail/item_%d_thumbnail.jpg", false, false, %d, 3);' 
          % (df['org_price'].iloc[i], df['dis_price'].iloc[i], df['name'].iloc[i], df['item_id'].iloc[i], random.randint(1, 4)))
    print()
"""

"""
for i in range(50):
    print('INSERT INTO item_img (org_filename, stored_filename, url, file_size, item_id)')
    print('VALUES (\'%s\', \'item_%d_thumbnail\', "https://commerce-side.s3.ap-northeast-2.amazonaws.com/items/thumbnail/item_%d_thumbnail.jpg", 30, %d);'
          % (df['name'].iloc[i], df['item_id'].iloc[i], df['item_id'].iloc[i], df['item_id'].iloc[i]))
    print()
"""

"""
for i in range(50):
    print('INSERT INTO item_description (type, url, text, seq, item_id)')
    print('VALUES (\'image\', "https://commerce-side.s3.ap-northeast-2.amazonaws.com/items/description/%d/item_%d_desc_1.jpg", null, 1, %d);'
          % (df['item_id'].iloc[i], df['item_id'].iloc[i], df['item_id'].iloc[i]))
    print()
"""


f = open('insert_item_option.txt', 'w')

for i in range(3,301):
      f.write('INSERT INTO item_option (description, plus_price, stock, item_id)\n')
      f.write('VALUES (\'Original Edition\', 0, %d, %d);\n' % (random.randint(50,200), i))
      f.write('\n')
      
      f.write('INSERT INTO item_option (description, plus_price, stock, item_id)\n')
      f.write('VALUES (\'Special Edition\', 1500, %d, %d);\n' % (random.randint(50,200), i))
      f.write('\n')

f.close()