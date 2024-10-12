t1 = """duğumuzu bilmek istiyorum. Bir dakikalığına volta atmayı keser misin?”\nHücrenin ortasında durup kaşlarımı kaldırarak ona bakıyorum.\n“Kusura bakma,” diye geveliyor ağzının içinde.\n“Sorun değil,” diyor Christina. “Burada fazla kaldık.”\nEvelyn’in birkaç küçük emirle Bilgelik Merkezi’nin lobisinde kaos yaratarak bütün tutsakların üçüncü kattaki hücrelerine kaçışmasını sağlamasının üzerinden günler geçti. Yaralarımızı tedavi etmek ve ağrı kesici dağıtmak için Topluluksuz bir kadın geldi. Karnımızı doyurduk, birçok kez duş aldık ama dı- şarıda neler olduğuna dair kimse bize bir şey söylemedi. Kimse inatçı sorularımı yanıtlamadı.\n“Tobias şimdiye kadar gelir sanıyordum,” diyorum, karyolamın kenarına çökerek. “Nerede kaldı?”\n“Belki yalan söylediğin ve babasıyla arkasından iş çevirdiğin için hâlâ sana kızgındır,” diyor Cara.\nAteş saçan gözlerle ona bakıyorum.\n“Dört, o kadar da dar kafalı biri değil,” diyor Christina. Ya Cara’yı azarlıyor ya da beni rahatlatmaya çalışıyor, emin olamıyorum. “Muhtemelen gelmesini engelleyen bir şeyler vardır. Ona güvenmeni söylemişti.”\nHerkes bağırıp çağırırken ve Topluluksuzlar bizi merdivenlere doğru ittirirken yaşanan kaosta, onu kaybetmemek için parmağımı tişörtüne dolamıştım. Bileğimi tutmuş, beni itmiş ve gerçekten de bu kelimeleri söylemişti. Güven bana. Sana söyledikleri yere git.\n“Güvenmeye çalışıyorum,” derken ciddiyim. Ona güvenmeye çalışıyorum. Ama her bir parçam, her bir kasım, her bir sinir ucum özgürlük için karıncalanıyor. Sadece bu hücreden değil, bizi tutsak eden şehirden kurtulmak istiyorum.\nÇitin dışında ne olduğunu görmem lazım.\nİKİNCİ BÖLÜM TOBIAS\nBu KORİDORLARDA YÜRÜRKEN, BİR TUTSAK OLARAK YALIN AYAK attığım her adımda acının her yerimde zonkladığı günleri hatırlamadan edemiyorum. Ve bu anılara, başka anılar eşlik ediyor. Ölüme giderken Beatrice Prior’u beklemek, kapıyı döven yumruklarım, sadece uyuşturucuyla ba"""
t2 = """yıltıldığını söyleyen Peter’ın kucağındayken Tris’in jöle gibi sarkan bacakları...\nBuradan nefret ediyorum.\nBilgelik yerleşkesi olduğu günlerdeki kadar temiz değil. Duvarlardaki kurşun delikleri, her yere yayılmış kırık ampul parçalarıyla tam bir savaş harabesine dönüşmüş durumda. Çamurlu ayakkabı izleri üzerinden hücresine doğru yürürken üzerimdeki ışıklar göz kırpıp duruyor. Ve sorgusuz sualsiz içeri alındım, çünkü kolumdaki siyah renkli bantta Topluluksuzlar’ın sembolü -boş bir daire- yüzümdeyse Evelyrnin yüz hatları var. Tobias Eaton, bir zamanlar utancın ismiydi, şimdiyse alabildiğine güçlü bir isim.\nTris içeride. Cara’nın karşısında Christina’yla omuz omuza vermiş yerde oturuyor. Sevgili Tris’"""
import FuncHub
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

t1t = torch.tensor(FuncHub.tokenize(t1), dtype=torch.float32)
t2t = torch.tensor(FuncHub.tokenize(t2), dtype=torch.float32).T

max_length = max(len(t1t), len(t2t))
t1t_padded = torch.nn.functional.pad(t1t, (0, max_length - len(t1t)))
t2t_padded = torch.nn.functional.pad(t2t, (0, max_length - len(t2t)))

def print_matrix(matrix):
    for row in matrix:
        for element in row:
            print(element, end=' ')
        print()  # Newline after each row


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
output = cos(t1t_padded, t2t_padded)
print(output)

vect = TfidfVectorizer(min_df=1, stop_words="english") 
t1t = vect.fit_transform([t1,t2," ensesinde bitince hâlâ şaşırıyorum. Saçlarını kestiğinde mutlu olmuştum, çünkü bu saç kesimi cici kızlar için değil, savaşçı kızlar içindi ve buna ihtiyacı olduğundan emindim.\n“İçeri nasıl girdin?” diye soruyor yumuşacık, berrak sesiyle.\n“Benim adım Tobias Eaton,” dediğimde gülüyor.\n“Doğru. Sürekli unutuyorum.” Bana bakmak için biraz geriliyor. Rüzgârda dağılabilecek bir yaprak yığınıymış gibi bana bakarken gözlerinde bocalayan bir ifade var. “Neler oluyor? Neden bu kadar geç kaldın?”\nÇaresizce, yakarırcasına soruyor. Burada, bende bıraktığı korkunç anılar, onunkilerin yanında hiç kalır. İdamına yürüyüşü, ağabeyinin ihaneti, korku serumu... Onu buradan çıkarmalıyım.\nCara başını kaldırıp merakla bakıyor. Artık derime sığamıyormuşum gibi rahatsız hissediyorum. Birinin bana bakmasından nefret ediyorum.\n'Evelyn, şehri kontrolü altına aldı,” diyorum. “Kimse ondan izin almadan adımını atamıyor. Birkaç gün önce, zalimlere karşı birleşmemiz gerektiğine dair bir konuşma yaptı. Dışarıdaki insanlardan bahsediyor.”\n“Zalimler mi?” diyor Christina. Cebinden küçük bir şişe çıkarıp içindekini ağzına boşaltıyor -sanırım bacağındak"])
print((t1t*t1t.T)[2,1])
print_matrix((t1t*t1t.T).toarray())
