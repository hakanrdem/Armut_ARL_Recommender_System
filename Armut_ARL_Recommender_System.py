# Association Rule Based Recommender System

# İş Problemi

"""
Türkiye’nin en büyük online hizmet platformu olan Armut,
hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır. Bilgisayarın veya akıllı
telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve
kategorileri içeren veri setini kullanarak Association Rule Learning ile ürün
tavsiye sistemi oluşturulmak istenmektedir.

"""
"""
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

"""
# Proje Görevleri

# Görev 1
# Veriyi Hazırlama

# Adım 1
# armut_data.csv dosyasını okutunuz.

!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/4_Tavsiye_Sistemleri/Armut_ARL_Recommender_System/armut_data.csv")
df = df_.copy()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# Adım 3:
# Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı
# (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek için bir sepet
# (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin
# aylık aldığı hizmetlerdir. Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı
# 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7
# hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması
# gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir
# değişkene atayınız.

df["CreateDate"].dtypes # > datetime olmalıdır.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

# Görev 2
# Birliktelik Kuralları Üretiniz ve Öneride bulununuz

# Adım 1
# Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

 # sepetID_Hizmet_df = df.groupby(['SepetID', 'Hizmet']).agg({"Hizmet": "count"}). \
                    # unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

sepetID_Hizmet_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Adım 2
# Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(sepetID_Hizmet_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

# Adım 3
# arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 10)

# Out[60]: ['2_0', '25_0', '9_4', '38_4', '15_1']