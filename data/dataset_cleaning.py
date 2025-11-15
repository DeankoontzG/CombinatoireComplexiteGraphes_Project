from operator import le
import pandas as pd
import unidecode
import os

input_file = 'Lexique383.xlsb'
output_file = 'lexique_filtre_cleaned.parquet'
ratio_selection = 0.15

print(f"Chargement du fichier {input_file}...")

df = pd.read_excel(input_file, engine='pyxlsb')
df = df.rename(columns={'1_ortho': 'mot', '15_nblettres': 'longueur', 'freqToutCourt': 'frequence'})
df = df[['mot', 'longueur', 'frequence']].drop_duplicates(subset=['mot']).copy()
df['mot'] = df['mot'].astype(str).str.lower()

# Nettoyage chars problématiques
chars_a_supprimer = [' ', '-', '_', "'"]

for char in chars_a_supprimer:
    nb_supprime = df['mot'].str.count(char)
    df['mot'] = df['mot'].str.replace(char, '', regex=False)
    df['longueur'] -= nb_supprime

df['mot_sans_accent'] = df['mot'].apply(lambda x: unidecode.unidecode(x))
df['mot'] = df['mot_sans_accent']
df = df.drop(columns=['mot_sans_accent'])

# On réduit le dico, en sélectionnant les 15% des mots les plus fréquents par taille

grouped = df.groupby('longueur')
mots_selectionnes_df = []

print(f"Début du filtrage : sélection des {ratio_selection*100:.0f}% plus fréquents par taille.")

for length, group in grouped:
    if length > 1 and length <= 12 :
        group_sorted = group.sort_values(by='frequence', ascending=False)
        nb_a_conserver = min(int(len(group_sorted)),max(2000, int(len(group_sorted) * ratio_selection))) 
        mots_selectionnes = group_sorted.head(nb_a_conserver)
        mots_selectionnes_df.append(mots_selectionnes)
    else : 
        group_sorted = group.sort_values(by='frequence', ascending=False)
        nb_a_conserver = 0

    print(f"  - Longueur {length}: {len(group_sorted)} mots -> {nb_a_conserver} conservés.")

df_final = pd.concat(mots_selectionnes_df, ignore_index=True)
df_final = df_final.drop_duplicates(subset=['mot']).sort_values(by=['longueur', 'frequence'], ascending=[True, False])

print(f"\nNombre total de mots uniques conservés : {len(df_final)}")

df_final.to_parquet(output_file, index=False)

print(f"Fichier Parquet sauvegardé sous : {output_file}")