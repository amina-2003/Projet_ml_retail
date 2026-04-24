
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether)
import os
from PIL import Image as PILImage

# ── Palette ──────────────────────────────────────────────────────
NAVY    = colors.HexColor('#0F172A')
BLUE    = colors.HexColor('#2563EB')
LBLUE   = colors.HexColor('#EFF6FF')
BORDER  = colors.HexColor('#CBD5E1')
GRAY    = colors.HexColor('#64748B')
LGRAY   = colors.HexColor('#F1F5F9')
RED     = colors.HexColor('#DC2626')
GREEN   = colors.HexColor('#16A34A')
ORANGE  = colors.HexColor('#D97706')
WHITE   = colors.white

W, H = A4
IMG = '/'

def S(name, **kw): return ParagraphStyle(name, **kw)

Title   = S('T',   fontName='Helvetica-Bold',   fontSize=26, textColor=WHITE, alignment=TA_CENTER)
Sub     = S('Su',  fontName='Helvetica',          fontSize=12, textColor=colors.HexColor('#BFDBFE'), alignment=TA_CENTER)
Tag     = S('Ta',  fontName='Helvetica',          fontSize=10, textColor=colors.HexColor('#93C5FD'), alignment=TA_CENTER)
H1      = S('H1',  fontName='Helvetica-Bold',   fontSize=15, textColor=NAVY,  spaceBefore=12, spaceAfter=5)
H2      = S('H2',  fontName='Helvetica-Bold',   fontSize=11, textColor=BLUE,  spaceBefore=8,  spaceAfter=4)
H3      = S('H3',  fontName='Helvetica-Bold',   fontSize=9.5,textColor=NAVY,  spaceBefore=5,  spaceAfter=3)
Body    = S('B',   fontName='Helvetica',          fontSize=9,  textColor=NAVY,  leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
Small   = S('Sm',  fontName='Helvetica',          fontSize=8,  textColor=GRAY,  leading=12, spaceAfter=3)
Bullet  = S('Bu',  fontName='Helvetica',          fontSize=9,  textColor=NAVY,  leading=13, leftIndent=12, spaceAfter=2)
Caption = S('Ca',  fontName='Helvetica-Oblique',  fontSize=8,  textColor=GRAY,  alignment=TA_CENTER, spaceAfter=8)
Formula = S('Fo',  fontName='Courier',            fontSize=8.5,textColor=NAVY,  backColor=LGRAY, leftIndent=12, rightIndent=12, spaceBefore=3, spaceAfter=3)
Note    = S('No',  fontName='Helvetica-Oblique',  fontSize=8.5,textColor=ORANGE,leftIndent=10, spaceAfter=3)
Tip     = S('Ti',  fontName='Helvetica',           fontSize=8.5,textColor=GREEN, leftIndent=10, spaceAfter=3)
Fix     = S('Fi',  fontName='Helvetica',           fontSize=9,  textColor=GREEN, leftIndent=12, spaceAfter=2)
Verdict = S('Ve',  fontName='Helvetica-BoldOblique',fontSize=9, textColor=GREEN,
            backColor=colors.HexColor('#F0FDF4'), leftIndent=10, rightIndent=10, spaceBefore=5, spaceAfter=5, leading=14)

def sp(n=0.3): return Spacer(1, n*cm)
def hr(): return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=5, spaceBefore=3)

def img(name, width=14*cm):
    path = os.path.join(IMG, name)
    if not os.path.exists(path): return sp(0.2)
    with PILImage.open(path) as im: w,h = im.size
    return Image(path, width=width, height=width*(h/w))

def img2(name, width=5.3*cm):
    path = os.path.join(IMG, name)
    if not os.path.exists(path): return sp(0.1)
    with PILImage.open(path) as im: w,h = im.size
    return Image(path, width=width, height=width*(h/w))

def section_hdr(title, sub=None):
    items = []
    t = Table([[Paragraph(title, S('sh', fontName='Helvetica-Bold', fontSize=13, textColor=WHITE))]], colWidths=[17*cm])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),BLUE),
        ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    items.append(t)
    if sub: items.append(Paragraph(sub, S('ss', fontName='Helvetica-Oblique', fontSize=9, textColor=GRAY, spaceAfter=5, spaceBefore=2)))
    return items

def kpi_row(items):
    """items = [(label, value, note, color)]"""
    n = len(items)
    cw = [17*cm/n]*n
    rows = [
        [Paragraph(l, S('kl', fontName='Helvetica', fontSize=8, textColor=GRAY, alignment=TA_CENTER)) for l,v,no,c in items],
        [Paragraph(v, S('kv', fontName='Helvetica-Bold', fontSize=18, textColor=c, alignment=TA_CENTER)) for l,v,no,c in items],
        [Paragraph(no, S('kn', fontName='Helvetica', fontSize=7.5, textColor=GRAY, alignment=TA_CENTER)) for l,v,no,c in items],
    ]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LBLUE),
        ('BOX',(0,0),(-1,-1),0.5,BORDER),('LINEAFTER',(0,0),(-2,-1),0.5,BORDER),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    return t

def grid_table(data, col_widths, header_bg=NAVY, row_alt=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND',(0,0),(-1,0),header_bg),('TEXTCOLOR',(0,0),(-1,0),WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
        ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]
    if row_alt: style.append(('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]))
    t.setStyle(TableStyle(style))
    return t

# ════════════════════════════════════════════════════════
story = []
doc = SimpleDocTemplate('rapport_churn_retail.pdf', pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=2*cm)

# ─── COVER ──────────────────────────────────────────────
cover = Table([
    [Paragraph('RAPPORT DE PROJET ML', Tag)],
    [Paragraph('Prédiction du Churn Client Retail', Title)],
    [Paragraph('XGBoost · Random Forest · KMeans · Flask API', Sub)],
    [sp(0.4)],
    [Paragraph('Pipeline Machine Learning complet — 4 372 clients × 52 features', 
        S('ct', fontName='Helvetica', fontSize=9.5, textColor=colors.HexColor('#7DD3FC'), alignment=TA_CENTER))],
], colWidths=[17*cm])
cover.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#1E3A8A')),
    ('TOPPADDING',(0,0),(-1,-1),16),('BOTTOMPADDING',(0,0),(-1,-1),16),
    ('LEFTPADDING',(0,0),(-1,-1),20),('RIGHTPADDING',(0,0),(-1,-1),20)]))
story.append(cover); story.append(sp(0.7))

# KPI banner — latest v3 results
story.append(kpi_row([
    ('AUC-ROC', '0.8953', 'XGBoost Optuna v3', BLUE),
    ('F1-Score', '0.7392', 'seuil par défaut 0.5', BLUE),
    ('F1 Optimal', '0.760', 'seuil = 0.389', GREEN),
    ('Recall', '85.2%', 'au seuil 0.389', GREEN),
    ('AP Score', '0.863', 'Precision-Recall', BLUE),
    ('Overfitting', '0.184', 'gap Train-Test ↓', ORANGE),
]))
story.append(sp(0.5))

# ─── Résumé exécutif ───────────────────────────────────
story.extend(section_hdr('Résumé Exécutif'))
story.append(sp(0.2))
story.append(Paragraph(
    'Ce rapport présente un pipeline complet de prédiction du churn pour un retailer e-commerce. '
    'À partir de <b>4 372 clients × 52 features</b>, le système enchaîne préprocessing, segmentation '
    'KMeans, entraînement de trois modèles (Logistic Regression, Random Forest, XGBoost), '
    'double optimisation (GridSearchCV + Optuna v3 avec early stopping), et déploiement Flask.', Body))
story.append(Paragraph(
    'La version v3 du pipeline introduit deux améliorations majeures : <b>early stopping natif XGBoost</b> '
    '(30 rounds) et <b>hyperparamètres contraints</b> (min_child_weight ∈ [5,15], gamma ∈ [1,5]) '
    'qui réduisent le gap d\'overfitting de <b>0.257 → 0.184</b>. '
    'Le modèle final atteint <b>AUC-ROC = 0.8953</b> et détecte <b>85.2% des churners</b> au seuil 0.389.', Body))
story.append(sp(0.3))

# ToC
toc_data = [['§','Section','Page'],
    ['1','Analyse Exploratoire (EDA)','2'],['2','Pipeline de Préprocessing','3'],
    ['3','Segmentation KMeans','4'],['4','Entraînement & Comparaison des Modèles','5'],
    ['5','Courbes ROC & Precision-Recall','6'],['6','Matrices de Confusion','7'],
    ['7','Importance des Features','8'],['8','Définition & Interprétation des Métriques','9'],
    ['9','Analyse de l\'Overfitting','10'],['10','Conclusions & Recommandations Business','11']]
toc = Table([[Paragraph(r[0],Small),Paragraph(r[1],Small),Paragraph(r[2],Small)] for r in toc_data],
    colWidths=[0.8*cm,14.5*cm,1.2*cm])
toc.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
    ('LINEBELOW',(0,1),(-1,-1),0.2,BORDER),('ALIGN',(2,0),(2,-1),'RIGHT')]))
story.append(toc); story.append(PageBreak())

# ─── SECTION 1 — EDA ────────────────────────────────────
story.extend(section_hdr('1. Analyse Exploratoire des Données (EDA)',
    '4 372 clients · 52 features brutes · 33.3% de churners'))
story.append(sp(0.2))
story.append(Paragraph('<b>1.1 Déséquilibre des classes</b>', H2))
story.append(Paragraph(
    'Le dataset présente un ratio <b>2.006:1</b> (2 918 non-churners vs 1 454 churners). '
    'Sans traitement, un classifieur naïf prédit toujours "Non-Churn" et atteint 66.7% d\'accuracy '
    'sans détecter aucun churner. Ce déséquilibre est corrigé par <b>SMOTE</b> sur le train set '
    'et <b>scale_pos_weight dynamique</b> dans XGBoost.', Body))
story.append(img('eda_churn_distribution.png', 14*cm))
story.append(Paragraph('Fig. 1 — Distribution Churn/Non-Churn (gauche: proportions, droite: décompte)', Caption))

story.append(Paragraph('<b>1.2 Distributions des variables clés par classe</b>', H2))
story.append(Paragraph(
    'Les histogrammes superposés révèlent les signaux discriminants : les churners ont '
    'une <b>Frequency plus faible</b>, des <b>AccountAgeDays plus bas</b> (clients récents), '
    'et un profil RFM différent. AvgDaysBetweenPurchases plus élevé indique un désengagement progressif.', Body))
story.append(img('eda_distributions.png', 16.5*cm))
story.append(Paragraph('Fig. 2 — Distributions comparées pour 8 features clés (Churn=rouge, Non-Churn=bleu)', Caption))
story.append(PageBreak())

# ─── SECTION 2 — PREPROCESSING ──────────────────────────
story.extend(section_hdr('2. Pipeline de Préprocessing v3',
    '8 étapes · 52 → 48 features · seuil multicolinéarité abaissé à 0.85'))
story.append(sp(0.2))

steps = [
    ('1. Nettoyage','18 colonnes supprimées : data leakage (ChurnRiskCategory, CustomerType, RFMSegment…), '
     'redondances (UniqueInvoices=Frequency, CancelledTransactions=NegativeQuantityCount). '
     'Clip : SatisfactionScore [0,10], SupportTicketsCount [0,+∞]. 79 NaN dans AvgDaysBetweenPurchases.'),
    ('2. Parsing dates','RegistrationDate : 7 formats détectés (ISO, DD/MM/YYYY, MM/DD/YYYY, ambigus → convention UK). '
     '0 NaT. Nouvelles features : RegMonth, RegWeekday, RegIsWeekend, AccountAgeDays, IsNewClient, RegSeason.'),
    ('3. Feature Eng.','5 nouvelles features : AvgBasketValue, CancellationRate, SpendingVolatility, EngagementScore, '
     'ProductsPerTransaction. TransactionsPerDay et MonetaryPerDay supprimées (nécessitent des colonnes leakage).'),
    ('4. Imputation','Médiane pour numériques (AvgDaysBetweenPurchases). Mode pour catégorielles. '
     'KNNImputer (k=5) sur SatisfactionScore et AvgDaysBetweenPurchases si encore manquants.'),
    ('5. Encodage','OrdinalEncoder : SpendingCategory [Low→VIP], BasketSizeCategory [Petit→Grand]. '
     'TargetEncoder : Region (smoothing=1.0, fit sur train uniquement). OHE : 6 colonnes nominales → 58 features.'),
    ('6. Multicolinéarité','Seuil abaissé à 0.85 (vs 0.90 v1) → 10 colonnes supprimées. '
     'Nouvelles suppressions : UniqueDescriptions (r=0.878 avec TotalTransactions), TotalQuantity (r=0.900 avec MonetaryTotal).'),
    ('7. Normalisation','RobustScaler sur 25 colonnes continues. 23 colonnes binaires conservées. '
     'Résistant aux outliers (médiane + IQR) vs StandardScaler (moyenne + std).'),
    ('8. SMOTE','Rééquilibrage train : {0:2334, 1:1163} → {0:2334, 1:2334}. '
     'Appliqué APRÈS le split. Note : idéalement à intégrer dans les folds CV (imblearn Pipeline).'),
]
for title, desc in steps:
    row = Table([[Paragraph(title, H3), Paragraph(desc, Small)]],
        colWidths=[3.5*cm, 13.5*cm])
    row.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0),LBLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),7),('GRID',(0,0),(-1,-1),0.3,BORDER)]))
    story.append(row); story.append(sp(0.1))

story.append(sp(0.2))
story.append(Paragraph('⚠️ Le SMOTE est appliqué sur l\'ensemble du train avant la CV dans Optuna → les folds de validation '
    'contiennent des données synthétiques proches des données réelles, ce qui surestime légèrement le CV score. '
    'Solution recommandée : <b>imblearn.pipeline.Pipeline</b> pour confiner le SMOTE dans chaque fold d\'entraînement.', Note))
story.append(PageBreak())

# ─── SECTION 3 — KMEANS ─────────────────────────────────
story.extend(section_hdr('3. Segmentation KMeans',
    'Clustering non supervisé · Méthode Elbow + Silhouette · Features RFM'))
story.append(sp(0.2))
story.append(Paragraph(
    'Le clustering KMeans segmente les clients <b>sans utiliser le label Churn</b>, '
    'révélant des groupes comportementaux homogènes. La feature <b>Cluster</b> est ensuite '
    'injectée dans les modèles supervisés comme signal d\'appartenance.', Body))

story.append(Paragraph('<b>3.1 Sélection du nombre optimal de clusters</b>', H2))
story.append(Paragraph(
    '<b>Méthode Elbow :</b> on trace l\'inertie (SSE = somme des distances au centroïde) en fonction de k. '
    'Le "coude" indique le k au-delà duquel l\'amélioration devient marginale. '
    '<b>Score Silhouette</b> s(i) ∈ [-1,1] : mesure la cohésion intra-cluster vs séparation inter-clusters. '
    'Un score proche de 1 signifie que les clusters sont bien séparés et compacts.', Body))
story.append(img('clustering_choix_k.png', 15*cm))
story.append(Paragraph('Fig. 3 — Elbow (gauche) et Silhouette (droite) pour k ∈ [2,10]', Caption))

story.append(Paragraph('<b>3.2 Visualisation PCA 2D et profil des clusters</b>', H2))
story.append(img('clustering_pca_2d.png', 15*cm))
story.append(Paragraph('Fig. 4 — Projection PCA 2D : clusters (gauche) et label Churn (droite)', Caption))

t2 = Table([[img2('clustering_churn_rate.png',8*cm), img2('clustering_profile_heatmap.png',8*cm)]], colWidths=[8.5*cm, 8.5*cm])
t2.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
story.append(t2)
story.append(Paragraph('Fig. 5 — Taux de churn par cluster (gauche) · Heatmap profil normalisé (droite)', Caption))
story.append(Paragraph('💡 <b>Usage business :</b> Les clusters à fort taux de churn sont les cibles prioritaires '
    'des campagnes de rétention. La feature Cluster capture un niveau de risque structurel non linéaire.', Tip))
story.append(PageBreak())

# ─── SECTION 4 — MODÈLES ────────────────────────────────
story.extend(section_hdr('4. Entraînement & Comparaison des Modèles',
    'Seuil de décision = 0.389 · SMOTE · scale_pos_weight dynamique'))
story.append(sp(0.2))

story.append(Paragraph('<b>4.1 Résultats comparatifs (seuil optimal 0.389)</b>', H2))
res_data = [
    ['Modèle','Accuracy','Precision','Recall ⭐','F1-Score','AUC-ROC','CV F1'],
    ['Logistic Regression','70.3%','53.2%','88.7%','0.6649','0.8453','0.8257'],
    ['Random Forest','72.7%','55.6%','89.0%','0.6843','0.8738','0.8529'],
    ['XGBoost (base)','81.0%','67.9%','81.4%','0.7406','0.8927','0.8561'],
    ['XGBoost (GridSearch)','80.0%','65.2%','85.6%','0.7400','0.8919','0.8651'],
    ['XGBoost (Optuna v3) ★','80.0%','65.3%','85.2%','0.7392','0.8953','0.8612'],
]
cw=[4.5*cm,2*cm,2*cm,2*cm,1.9*cm,1.9*cm,1.9*cm]
rt=Table(res_data,colWidths=cw)
rt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),
    ('BACKGROUND',(0,5),(-1,5),colors.HexColor('#DCFCE7')),
    ('FONTNAME',(0,5),(-1,5),'Helvetica-Bold'),
    ('TEXTCOLOR',(0,5),(-1,5),colors.HexColor('#15803D')),
    ('BACKGROUND',(3,0),(3,0),colors.HexColor('#1E40AF')),
]))
story.append(rt)
story.append(Paragraph('Tab. 1 — Tableau comparatif complet (★ = modèle retenu)', Caption))

story.append(img('metrics_comparison_barplot.png', 15*cm))
story.append(Paragraph('Fig. 6 — Comparaison visuelle des 5 métriques au seuil 0.389', Caption))

story.append(Paragraph('<b>4.2 Améliorations v3 (Optuna)</b>', H2))
improv = [
    'Early stopping natif XGBoost (early_stopping_rounds=30) avec eval set 15% → arrêt automatique avant overfitting',
    'max_depth contraint à [3,6] au lieu de [2,8] → arbres moins profonds',
    'min_child_weight contraint à [5,15] → exige plus d\'observations par feuille',
    'gamma contraint à [1,5] → pruning plus agressif des arbres',
    'Optuna optimise sur roc_auc (plus stable que F1 sur classes déséquilibrées)',
    'scale_pos_weight calculé dynamiquement : n_neg/n_pos (vs valeur fixe 2)',
]
for item in improv:
    story.append(Paragraph(f'✅ {item}', Fix))
story.append(PageBreak())

# ─── SECTION 5 — ROC & PR ───────────────────────────────
story.extend(section_hdr('5. Courbes ROC & Precision-Recall',
    'Évaluation à tous les seuils de décision · AP = 0.863'))
story.append(sp(0.2))

story.append(Paragraph('<b>5.1 Comparaison ROC vs Precision-Recall</b>', H2))
comp = [
    ['','Courbe ROC','Courbe Precision-Recall'],
    ['Axes','FPR (x) vs TPR/Recall (y)','Recall (x) vs Precision (y)'],
    ['Métrique synthèse','AUC-ROC — aire sous courbe','AP — Average Precision (aire)'],
    ['Référence aléatoire','AUC = 0.50','AP = taux positifs = 0.333'],
    ['Avantage','Vision globale de la discrimination','Robuste au déséquilibre ← notre cas'],
    ['Nos résultats XGBoost','AUC = 0.8953 ★','AP = 0.863 ★'],
]
ct=Table(comp,colWidths=[3.5*cm,6.5*cm,7*cm])
ct.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('BACKGROUND',(0,1),(0,-1),LGRAY),('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
    ('FONTSIZE',(0,0),(-1,-1),8.5),('ROWBACKGROUNDS',(1,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('BACKGROUND',(2,5),(-1,5),colors.HexColor('#DCFCE7')),
    ('FONTNAME',(2,5),(-1,5),'Helvetica-Bold'),
    ('BACKGROUND',(2,4),(-1,4),colors.HexColor('#FEF9C3')),
]))
story.append(ct); story.append(sp(0.3))

story.append(img('roc_pr_curves_comparison.png', 16.5*cm))
story.append(Paragraph('Fig. 7 — Courbes ROC (gauche) et PR (droite) — comparaison des 3 modèles', Caption))

story.append(Paragraph('<b>5.2 Courbe PR finale et seuil optimal (XGBoost Optuna v3)</b>', H2))
story.append(Paragraph(
    'La courbe PR relie tous les couples (Recall, Precision) obtenus en faisant varier le seuil de décision '
    'de 0 à 1. L\'<b>Average Precision (AP=0.863)</b> est l\'aire sous cette courbe, pondérée par les '
    'variations de Recall. Le <b>seuil optimal F1 = 0.389</b> maximise l\'équilibre Precision/Recall.', Body))
story.append(img('precision_recall_final.png', 13*cm))
story.append(Paragraph('Fig. 8 — Courbe PR finale avec seuil optimal mis en évidence (★)', Caption))

story.append(Paragraph('<b>5.3 Impact du seuil de décision</b>', H2))
thresh_data=[
    ['Seuil','Precision','Recall','F1','Usage recommandé'],
    ['0.5 (défaut)','~75%','~72%','~0.735','Contexte équilibré — moins de fausses alertes'],
    ['0.389 (optimal F1) ★','72.0%','85.2%','0.760','Production — détection maximale churners'],
    ['0.285 (v2)','67.5%','86.3%','0.757','Très agressif — si coût FN >> coût FP'],
    ['0.20 (seuil bas)','~58%','~93%','~0.72','Marketing de masse — trop de FP'],
]
tht=Table(thresh_data,colWidths=[3.5*cm,2.2*cm,2.2*cm,2*cm,7.1*cm])
tht.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('BACKGROUND',(0,2),(-1,2),colors.HexColor('#DCFCE7')),
    ('FONTNAME',(0,2),(-1,2),'Helvetica-Bold'),
]))
story.append(tht); story.append(Paragraph('Tab. 2 — Impact du seuil sur les métriques', Caption))
story.append(PageBreak())

# ─── SECTION 6 — CONFUSION MATRICES ────────────────────
story.extend(section_hdr('6. Matrices de Confusion',
    'TN · FP · FN ⚠️ · TP — seuil = 0.389'))
story.append(sp(0.2))

cm_explain=[
    ['','Prédit Non-Churn (0)','Prédit Churn (1)'],
    ['Réel Non-Churn (0)','TN — Vrai Négatif\n✅ Client fidèle correctement identifié','FP — Faux Positif\n⚠️ Alerte inutile (coût : campagne de rétention)'],
    ['Réel Churn (1)','FN — Faux Négatif\n❌ Churner manqué → client perdu définitivement','TP — Vrai Positif\n✅ Churner détecté → intervention possible'],
]
cmt=Table(cm_explain,colWidths=[4*cm,6.5*cm,6.5*cm])
cmt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('BACKGROUND',(0,1),(0,-1),LGRAY),('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
    ('BACKGROUND',(1,1),(1,1),colors.HexColor('#DBEAFE')),
    ('BACKGROUND',(2,1),(2,1),colors.HexColor('#FEE2E2')),
    ('BACKGROUND',(1,2),(1,2),colors.HexColor('#FEE2E2')),
    ('BACKGROUND',(2,2),(2,2),colors.HexColor('#DCFCE7')),
    ('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),0.5,BORDER),
    ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ('LEFTPADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
]))
story.append(cmt); story.append(sp(0.3))

# 3 matrices side by side
w_cm=5.2*cm
cm_row=Table([[img2('classification_confusion_matrix_lr.png',w_cm),
               img2('classification_confusion_matrix_rf.png',w_cm),
               img2('classification_confusion_matrix_xgb.png',w_cm)]],
    colWidths=[5.7*cm,5.7*cm,5.7*cm])
cm_row.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
story.append(cm_row)
lbl_row=Table([[Paragraph('Logistic Regression\nF1=0.659 | Recall=0.880',Caption),
                Paragraph('Random Forest\nF1=0.723 | Recall=0.735',Caption),
                Paragraph('XGBoost Optuna v3 ★\nF1=0.739 | Recall=0.852',Caption)]],
    colWidths=[5.7*cm,5.7*cm,5.7*cm])
story.append(lbl_row)
story.append(Paragraph('Fig. 9 — Matrices de confusion des 3 modèles au seuil 0.389', Caption))

story.append(Paragraph('<b>Décompte des erreurs — XGBoost Optuna v3 (875 clients test)</b>', H2))
err=[['Type','Approx.','Signification & coût business'],
    ['TP — Vrais Positifs','~248','Churners alertés → action rétention possible ✅'],
    ['FN — Faux Négatifs ⚠️','~43','Churners manqués → clients perdus sans intervention (coût élevé)'],
    ['TN — Vrais Négatifs','~452','Clients fidèles correctement ignorés ✅'],
    ['FP — Faux Positifs','~132','Non-churners alertés → campagne inutile (coût faible)'],
]
et=Table(err,colWidths=[4*cm,2*cm,11*cm])
et.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('BACKGROUND',(0,2),(-1,2),colors.HexColor('#FEE2E2')),('FONTNAME',(0,2),(-1,2),'Helvetica-Bold'),
    ('BACKGROUND',(0,4),(-1,4),colors.HexColor('#DCFCE7')),
]))
story.append(et)
story.append(PageBreak())

# ─── SECTION 7 — FEATURE IMPORTANCE ────────────────────
story.extend(section_hdr('7. Importance des Features',
    'Top 20 — XGBoost vs Random Forest'))
story.append(sp(0.2))
story.append(Paragraph(
    'L\'importance des features (Gain normalisé) mesure la contribution de chaque variable '
    'à la réduction de l\'impureté dans les arbres. Les features en rouge = Top 3.', Body))

fi_data=[['#','Feature','Importance','Interprétation business'],
    ['1','Frequency','0.1668','Fréquence d\'achat — signal RFM #1. Baisse = désengagement imminent'],
    ['2','AccountAgeDays','0.0642','Âge du compte. Clients récents = churners potentiels'],
    ['3','SpendingCategory','0.0625','Segment dépense [Low→VIP]. Low = risque maximal'],
    ['4','ProductDiversity_Explorateur','0.0488','Profil diversifié = fidélité comportementale'],
    ['5','WeekendPreference_Weekend','0.0486','Préférence weekend — pattern comportemental clé'],
    ['6','IsNewClient','0.0392','Client < 1 an = instable, taux de churn plus élevé'],
    ['7','TotalTransactions','0.0379','Volume total transactions — corrèle avec Frequency'],
    ['8','RegSeason_hiver','0.0372','Inscrit en hiver — pattern saisonnier'],
    ['9','AvgDaysBetweenPurchases','0.0280','Intervalle moyen entre achats — allongement = alerte'],
    ['10','Gender_Unknown','0.0279','Genre inconnu = profil incomplet = comportement atypique'],
]
fit=Table(fi_data,colWidths=[0.8*cm,5*cm,2.2*cm,9*cm])
fit.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),4),
    ('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),
    ('ALIGN',(0,0),(0,-1),'CENTER'),('ALIGN',(2,0),(2,-1),'CENTER'),
    ('BACKGROUND',(0,1),(-1,3),colors.HexColor('#FEF9C3')),
    ('FONTNAME',(0,1),(-1,3),'Helvetica-Bold'),
]))
story.append(fit); story.append(Paragraph('Tab. 3 — Top 10 features XGBoost v3 (jaune = Top 3)', Caption))

fi_row=Table([[img2('classification_feature_importance_rf.png',8*cm),
               img2('classification_feature_importance_xgb.png',8*cm)]],
    colWidths=[8.5*cm,8.5*cm])
fi_row.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')]))
story.append(fi_row)
story.append(Paragraph('Fig. 10 — Feature importance : Random Forest (gauche) vs XGBoost v3 (droite)', Caption))

story.append(Paragraph('<b>Changements par rapport à v2</b> : Frequency passe de la position 2 à <b>#1</b> '
    '(SpendingCategory était #1 en v2). IsNewClient apparaît au rang 6 (supprimée en v1 pour multicolinéarité '
    'mais réintégrée en v3). Ces changements reflètent le seuil de multicolinéarité abaissé de 0.90 à 0.85.', Small))
story.append(PageBreak())

# ─── SECTION 8 — MÉTRIQUES ──────────────────────────────
story.extend(section_hdr('8. Métriques d\'Évaluation — Définitions & Formules',
    'Accuracy · Precision · Recall · F1 · AUC-ROC · AP Score'))
story.append(sp(0.2))

met_data=[
    ['Métrique','Formule','Interprétation','Priorité'],
    ['Accuracy','(TP+TN)/(TP+TN+FP+FN)','% prédictions correctes — TROMPEUSE si déséquilibre (≠ utile)','⚠️'],
    ['Precision','TP / (TP+FP)','Parmi les alertes Churn émises, quelle fraction est réelle ?','⚠️'],
    ['Recall\n(Sensitivité)','TP / (TP+FN)','Parmi tous les churners, quelle fraction a été détectée ?','⭐⭐⭐'],
    ['F1-Score','2×P×R / (P+R)','Moyenne harmonique P/R — pénalise les déséquilibres P vs R','⭐⭐'],
    ['AUC-ROC','Aire sous ROC','P(score_churn > score_non_churn). 0.5=aléatoire, 1=parfait','⭐⭐'],
    ['AP Score','Σ P(k)·ΔR(k)','Aire sous courbe PR — robuste au déséquilibre des classes','⭐⭐'],
]
mett=Table(met_data,colWidths=[3*cm,4.5*cm,7.8*cm,1.7*cm])
mett.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(3,0),(3,-1),'CENTER'),
    ('FONTNAME',(1,1),( 1,-1),'Courier'),
    ('BACKGROUND',(0,3),(-1,3),colors.HexColor('#FEF9C3')),('FONTNAME',(0,3),(-1,3),'Helvetica-Bold'),
]))
story.append(mett); story.append(Paragraph('Tab. 4 — Métriques d\'évaluation avec priorités Churn (⭐ = priorité)', Caption))

story.append(Paragraph('<b>Formules complètes</b>', H2))
formulas=['Accuracy   = (TP + TN) / (TP + TN + FP + FN)',
    'Precision  = TP / (TP + FP)    ← fiabilité des alertes',
    'Recall     = TP / (TP + FN)    ← MAXIMISER — churners détectés',
    'F1-Score   = 2 × Precision × Recall / (Precision + Recall)',
    'AUC-ROC    = P( score(churn_réel) > score(non_churn_réel) )',
    'AP Score   = integral de Precision(Recall) d(Recall)  ≈  Σ P(k) × ΔR(k)']
for f in formulas: story.append(Paragraph(f, Formula))

story.append(Paragraph('<b>Coût asymétrique des erreurs en contexte Churn</b>', H2))
cost=[['Erreur','Impact','Coût estimé','Réponse'],
    ['Faux Négatif (FN)','Churner non détecté → client perdu','⭐⭐⭐ ÉLEVÉ\nPerte LTV complète','Baisser le seuil de décision'],
    ['Faux Positif (FP)','Non-churner alerté → campagne inutile','⭐ FAIBLE\nCoût d\'une action marketing','Acceptable si FN minimisés'],
]
costt=Table(cost,colWidths=[3.5*cm,4.5*cm,4*cm,5*cm])
costt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#FEE2E2')),('FONTNAME',(0,1),(-1,1),'Helvetica-Bold'),
    ('BACKGROUND',(0,2),(-1,2),colors.HexColor('#DCFCE7')),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),6),
    ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),6),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
]))
story.append(costt)
story.append(PageBreak())

# ─── SECTION 9 — OVERFITTING ────────────────────────────
story.extend(section_hdr('9. Analyse de l\'Overfitting',
    'Progression v1 → v2 → v3 · Gap 0.257 → 0.240 → 0.184'))
story.append(sp(0.2))

story.append(img('overfitting_analysis.png', 14*cm))
story.append(Paragraph('Fig. 11 — Train F1 vs Test F1 par modèle (Δ = gap overfitting)', Caption))

story.append(Paragraph('<b>Analyse détaillée — XGBoost Optuna v3</b>', H2))
ov_data=[['Métrique','Train','CV (mean)','CV (std)','Test','Verdict'],
    ['F1-Score','0.9246','0.8612','0.0110','0.7392','⚠️ Gap=0.184'],
    ['AUC-ROC','0.9813','—','—','0.8953','✅ Bon'],
    ['Train - CV','0.0634','—','—','—','⚠️ Mémorisation'],
    ['CV - Test','0.1203','—','—','—','⚠️ CV optimiste'],
]
ovt=Table(ov_data,colWidths=[4*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,3*cm])
ovt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('ALIGN',(1,0),(-1,-1),'CENTER'),
]))
story.append(ovt); story.append(Paragraph('Tab. 5 — Détail diagnostic overfitting v3', Caption))

story.append(Paragraph('<b>Progression par version</b>', H2))
prog=[['Version','Gap Train-Test','Cause principale','Correction appliquée'],
    ['v1','0.257','min_child_weight=1, SMOTE hors CV','Contraintes Optuna, RobustScaler'],
    ['v2','0.240','Optuna CV sur données SMOTE','scale_pos_weight dynamique, roc_auc dans Optuna'],
    ['v3 ★','0.184','Early stopping absent','Early stopping (30 rounds) + gamma/min_child_weight contraints'],
    ['Cible','< 0.10','SMOTE dans boucle CV','imblearn Pipeline (prochaine itération)'],
]
progt=Table(prog,colWidths=[2.5*cm,3*cm,5.5*cm,6*cm])
progt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('BACKGROUND',(0,3),(-1,3),colors.HexColor('#DCFCE7')),('FONTNAME',(0,3),(-1,3),'Helvetica-Bold'),
    ('BACKGROUND',(0,4),(-1,4),colors.HexColor('#FEF9C3')),
]))
story.append(progt)
story.append(Paragraph('Tab. 6 — Progression du gap d\'overfitting à travers les versions', Caption))

story.append(Paragraph('<b>Causes et solutions</b>', H2))
causes = [
    ('⚠️ SMOTE hors CV','Les folds de validation contiennent des données synthétiques similaires au train → CV score '
     'surestimé de ~0.12. Solution : <b>imblearn.pipeline.Pipeline</b> pour confiner SMOTE dans chaque fold d\'entraînement.'),
    ('⚠️ n_estimators=499','Trop d\'arbres permettent la mémorisation. L\'early stopping (30 rounds) a corrigé cela '
     'partiellement mais le modèle réentraîné utilise les 499 arbres sans arrêt anticipé.'),
    ('⚠️ CV score optimiste','Le gap CV-Test de 0.12 signifie que le CV voit des données SMOTE-augmentées '
     'proches du test réel mais non identiques. Solution : réparer le SMOTE dans CV.'),
]
for title, desc in causes:
    story.append(Paragraph(f'<b>{title}</b> — {desc}', Bullet))
story.append(PageBreak())

# ─── SECTION 10 — CONCLUSIONS ───────────────────────────
story.extend(section_hdr('10. Conclusions & Recommandations Business',
    'Synthèse technique · Plan d\'action · ROI estimé'))
story.append(sp(0.2))

story.append(kpi_row([
    ('AUC-ROC', '0.8953', 'XGBoost Optuna v3', BLUE),
    ('F1 Optimal', '0.760', 'seuil = 0.389', GREEN),
    ('Recall', '85.2%', '248/291 churners', GREEN),
    ('AP Score', '0.863', 'vs baseline 0.333', BLUE),
    ('Overfitting', '0.184', 'v1:0.257 → v3:0.184', ORANGE),
]))
story.append(sp(0.4))

story.append(Paragraph('<b>Synthèse technique</b>', H2))
tech=[['Composant','Choix technique','Justification'],
    ['Modèle','XGBoost (Optuna v3)','AUC-ROC=0.8953, meilleur F1 et recall'],
    ['Optimisation','Optuna TPE (100 trials) + early stopping','Bayesian > GridSearch, arrêt avant overfitting'],
    ['Déséquilibre','SMOTE + scale_pos_weight dynamique','Double protection, calcul ratio réel'],
    ['Normalisation','RobustScaler (médiane+IQR)','Résistant aux outliers vs StandardScaler'],
    ['Encodage Region','Target Encoding (smoothing=1.0)','Capture la corrélation région/churn sans OHE explosion'],
    ['Seuil production','0.389 (optimal F1)','Recall 85.2% vs 72% au seuil 0.5'],
    ['Clustering','KMeans (k optimal par Silhouette)','Feature Cluster enrichit les modèles supervisés'],
    ['Déploiement','Flask API (4 endpoints)','/health, /model/info, /predict, /predict/batch'],
    ['Threshold dual','0.5 (standard) + 0.285/0.389 (optimal)','API retourne les deux prédictions simultanément'],
]
techt=Table(tech,colWidths=[3.5*cm,5*cm,8.5*cm])
techt.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LGRAY]),
    ('GRID',(0,0),(-1,-1),0.3,BORDER),('TOPPADDING',(0,0),(-1,-1),5),
    ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
    ('VALIGN',(0,0),(-1,-1),'TOP'),
]))
story.append(techt); story.append(sp(0.3))

story.append(Paragraph('<b>Plan d\'action par priorité</b>', H2))
actions=[
    ('🔴 Immédiat','Déployer API Flask avec seuil 0.389 — /predict retourne déjà churn_prediction_optimal'),
    ('🔴 Immédiat','Cibler les clients dans le cluster à fort taux de churn pour campagnes de rétention'),
    ('🔴 Immédiat','Surveiller les clients avec Frequency en baisse sur 30 jours (feature #1)'),
    ('🟡 30 jours','Intégrer SMOTE dans imblearn Pipeline pour corriger le gap CV-Test de 0.12'),
    ('🟡 30 jours','Réentraîner XGBoost avec early_stopping_rounds sur le modèle final (pas seulement Optuna)'),
    ('🟡 30 jours','Mettre en place monitoring de data drift en production (PSI, KS test)'),
    ('🟢 60 jours','Lancer campagnes personnalisées selon segment Churn : SpendingCategory=Low + IsNewClient=1'),
    ('🟢 60 jours','Tester Stacking (LR + RF + XGBoost) comme meta-learner pour réduire variance'),
    ('🟢 90 jours','Ajouter SHAP values pour explications locales des churners détectés'),
    ('🟢 90 jours','Évaluer l\'ajout de features temporelles : évolution de Frequency sur 3/6/12 mois'),
]
for priority, action in actions:
    story.append(Paragraph(f'<b>{priority}</b> — {action}', Bullet))

story.append(sp(0.4)); story.append(hr())
story.append(Paragraph(
    '<b>ROI estimé :</b> Sur le test set (875 clients), le modèle détecte 248 churners sur 291 (Recall=85.2%). '
    'Hypothèses : LTV moyenne = 150€, taux reconquête campagne = 30%, coût campagne = 10€/client. '
    'ROI = (248 × 0.30 × 150) − (248 + 132) × 10 = <b>11 160 − 3 800 = +7 360 € net</b> par cycle sur ce seul test set. '
    'À l\'échelle du dataset complet (4 372 clients), l\'impact annuel est ~5× supérieur.', Verdict))

doc.build(story)
print("✅ rapport_churn_retail.pdf généré")

