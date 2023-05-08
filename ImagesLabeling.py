import json
import urllib.request
import pandas as pd
import os

# metadata links
url_metadata_page1 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=1&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='
url_metadata_page2 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=2&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='
url_metadata_page3 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=3&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='
url_metadata_page4 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=4&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='
url_metadata_page5 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=5&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='
url_metadata_page6 = 'https://www.gallerie-estensi.beniculturali.it/collezioni-digitali/docsearch?perpage=30&page=6&titolo=&autore=&oggetto=Moneta&localizzazione=&inventario=&materiaetecnica='

coin_dataset_path = "./Coins_Dataset/"


def digit_index(string):
    for i, c in enumerate(string):
        if c.isdigit():
            # print(i)
            return i


def check_cronologia(cronologia):
    cronologia = cronologia.replace('.', '')
    if '/' in cronologia:
        # print(cronologia.replace('/','-'))
        cronologia = cronologia.replace('(?)', '')
        cronologia = cronologia.replace('/', '-')
    if ',' in cronologia:
        cronologia = cronologia.replace(',', '')
    if 'ante' in cronologia:
        cronologia = cronologia.replace('ante', '')
    if 'post' in cronologia:
        cronologia = cronologia.replace('post', '')
    index = digit_index(cronologia)
    cronologia = '(' + cronologia[index:]
    return cronologia


def check_ambiti(ambito):
    return ambito.replace('Produzione', '')


def check_diametro(diametro):
    if '.' in diametro:
        return diametro.replace('.', ',')
    return diametro


def rename_images():
    # Downloading all metadata

    page = urllib.request.urlopen(url_metadata_page1).read()

    data = json.loads(page)
    elementi = data['Objs']

    """
    for d in elementi:
        print(d)
    """

    page2 = urllib.request.urlopen(url_metadata_page2).read()

    data2 = json.loads(page2)
    elementi2 = data2['Objs']

    page3 = urllib.request.urlopen(url_metadata_page3).read()

    data3 = json.loads(page3)
    elementi3 = data3['Objs']

    page4 = urllib.request.urlopen(url_metadata_page4).read()

    data4 = json.loads(page4)
    elementi4 = data4['Objs']

    page5 = urllib.request.urlopen(url_metadata_page5).read()

    data5 = json.loads(page5)
    elementi5 = data5['Objs']

    page6 = urllib.request.urlopen(url_metadata_page6).read()

    data6 = json.loads(page6)
    elementi6 = data6['Objs']

    lista = elementi + elementi2 + elementi3 + elementi4 + elementi5 + elementi6  # final result
    # print(len(lista))

    # create a dataframe to store data in another structure
    df = pd.DataFrame(columns=['id', 'ambiti', 'cronologia', 'diametro'])

    # dataset population
    for i in lista:
        dict_list = {'id': i['id'], 'ambiti': i['ambiti'], 'cronologia': i['cronologia'],'diametro': i['diametro']}
        # print(dict_list)
        tmp = pd.Series(dict_list)
        # print(tmp)
        df = df.append(tmp, ignore_index=True)

    """
    print ('unique')
    print (pd.unique(df['ambiti','cronologia']))
    """

    # label list
    label_list = []
    labels = []

    # rename files
    for imagepath in os.listdir(coin_dataset_path):
        # print(type(imagepath[2:7]))
        if digit_index(imagepath) != 0:
            if int(imagepath[2:7]) in list(df['id']):
                tmp = (df[df['id'] == int(imagepath[2:7])])
                tmp_cronologia = check_cronologia(str(tmp['cronologia'].iloc[0]))
                tmp_ambiti = check_ambiti(str(tmp['ambiti'].iloc[0]))
                tmp_diametro = check_diametro(str(tmp['diametro'].iloc[0]))
                # label list manager
                labels.append(tmp_ambiti + "_" + tmp_cronologia + "_" + tmp_diametro)
                if tmp_ambiti + "_" + tmp_cronologia + "_" + tmp_diametro in label_list:
                    # print('already present')
                    pass
                else:
                    label_list.append(tmp_ambiti + "_" + tmp_cronologia + "_" + tmp_diametro)
                new_path = (str(tmp['id'].iloc[0]) + "_" + tmp_ambiti + "_" + tmp_cronologia + "_" + tmp_diametro + ".jpg")
                # new_path = (str(tmp['id'].iloc[0]) +"_" +str(tmp['ambiti'].iloc[0]) +"_" + str(tmp['materia_tecnica'].iloc[0]) + '.jpg')
                old_file = coin_dataset_path + imagepath
                new_file = coin_dataset_path + new_path
                os.rename(old_file, new_file)
        else:
            string = imagepath[7:].replace('.jpg', '')
            # print(string)
            labels.append(string)
            if string in label_list:
                pass
            else:
                label_list.append(string)

    # calculate number of classes
    # print(len((df['cronologia'])))
    # print(len((df['ambiti'])))
    #
    # series_list = [
    #     df['cronologia'],
    #     df['ambiti'],
    #     df['diametro']
    # ]
    # print(len(pd.concat(series_list, axis=1, sort=False).sum(axis=1)))
    # print(len(pd.unique(pd.concat(series_list, axis=1, sort=False).sum(axis=1))))

    dict_label = dict((k, v) for k, v in zip(label_list, range(len(label_list))))
    # print(f'lenght label_list: {len(label_list)}')
    # print(f'dict: {dict_label}')
    # return dict with k,v for each class

    return dict_label, labels
