import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib.pyplot as plt
font_path=r'C:\Users\anous\Downloads\Roboto (1)\Roboto-Regular.ttf'
fm.fontManager.addfont(font_path)
font_prop=fm.FontProperties(fname=font_path)
plt.rcParams['font.family']=font_prop.get_name()
sns.set_palette('deep')