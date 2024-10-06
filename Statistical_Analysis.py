import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Statistics_Calculator:
    def __init__(self, Dataset):
        self.Dataset = Dataset

    def Sort_Price(self):  # Sắp xếp giá nhà trong khu vực theo thứ tự tăng dần
        return self.Dataset.sort_values(by="Price")

    def Count(self):  # Tính số lượng nhà trong khu vực
        return self.Dataset["Price"].shape[0]

    def Mean(self):  # Tính giá trị trung bình của giá nhà trong khu vực
        return self.Dataset["Price"].mean()

    def Standard_Deviation(self):  # Tính độ lệch chuẩn của giá nhà trong khu vực
        return math.sqrt(
            self.Dataset["Price"].apply(lambda x: (x - self.Mean()) ** 2).sum()
            / self.Count()
        )

    def Minimum(self):  # Tính giá trị nhỏ nhất của giá nhà trong khu vực
        return self.Dataset["Price"].min()

    def Maximum(self):  # Tính giá trị lớn nhất của giá nhà trong khu vực
        return self.Dataset["Price"].max()

    def Range(self):  # Tính phạm vi của giá nhà trong khu vực
        return self.Maximum() - self.Minimum()

    def Quantile(self, Mode):
        if Mode == str(25):  # Phân vị thứ 25%
            return self.Count() * 25 / 100
        elif Mode == str(50):  # Phân vị thứ 50%
            return self.Count() * 50 / 100
        elif Mode == str(75):  # Phân vị thứ 75%
            return self.Count() * 75 / 100
        elif Mode == "IQR":  #  Phạm vi tứ phân vị
            return self.Quantile(75) - self.Quantile(25)

    def Outliers(self, Mode):
        if Mode == "Lower":  # Phân tích ngoại lệ dưới
            return self.Quantile(25) - 1.5 * self.Quantile("IQR")
        if Mode == "Higher":  # Phân tích ngoại lệ trên
            return self.Quantile(75) + 1.5 * self.Quantile("IQR")

    def Mean_Absolute_Deviation(self):  # tính độ lệch tuyệt đối trung bình tuyệt đối
        return self.Dataset["Price"].apply(lambda x: abs(x - self.Mean())).mean()

    def Coefficient_of_Variation(self):  # Tính hệ số biến thiên
        return (self.Standard_Deviation() / self.Mean()) * 100

    def Correlation_Coefficient_Between_Area_and_Price(
        self,
    ):  # Hệ số tương quan giữa giá và diện tích
        Area_Mean = self.Dataset["Area"].mean()
        Price_Mean = self.Dataset["Price"].mean()
        Numerator = (
            (self.Dataset["Area"] - Area_Mean)
            * (self.Dataset["Price"] - Price_Mean)
        ).sum()
        Denominator = ((self.Dataset["Area"] - Area_Mean) ** 2).sum() * (
            (self.Dataset["Price"] - Price_Mean) ** 2
        ).sum()
        return Numerator / math.sqrt(Denominator)
    
    def Correlation_Coefficient_Between_Bedrooms_and_Price(self):
        return self.Dataset["Price"].corr(self.Dataset["Bedrooms"])


def Bar_Plot(House_List):
    plt.figure(figsize=(10, 6)) # Thiết lập kich thước biểu đồ
    sns.set_theme(style="whitegrid")  # Thiết lập theme sáng

    sns.barplot(x='Location', y='Price', data=House_List) # Vẽ biểu đồ cột với trục x là Location và trục y là Price
    
    plt.title('Giá trung bình bất động sản theo khu vực ở TP.HCM', fontsize=14)
    plt.xlabel('Khu vực', fontsize=12)
    plt.ylabel('Giá trung bình (tỷ đồng)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Tối ưu hóa không gian biểu đồ

    return plt.show()

def Box_Plot(House_List):
    plt.figure(figsize = (10, 6))
    sns.set_theme(style="whitegrid")

    sns.boxplot(x='Location', y='Price', data=House_List)
    
    plt.title("Phân phối giá ở các vùng khác nhau.", fontsize=14)
    plt.xlabel("Khu vực", fontsize=12)
    plt.ylabel("Giá (Tỷ VND)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Tối ưu hóa không gian biểu đồ

    return plt.show()

def Violin_Plot(House_List, Location):
    Location_Data = House_List[House_List['Location'] == Location]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.violinplot(x='Location', y='Price', data=Location_Data)

    plt.title(f"Phân tán giá ở {Location}", fontsize=14)
    plt.xlabel("Khu vực", fontsize=12)
    plt.ylabel("Giá", fontsize=12)
    plt.tight_layout()

    return plt.show()

def Scatter_Plot(House_List, Location):
    Location_Data = House_List[House_List['Location'] == Location]
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(x='Bedrooms', y='Price', data=Location_Data)

    plt.title("Phân tán giá", fontsize=14)
    plt.xlabel(f"Khu vực {Location}", fontsize=12)
    plt.ylabel("Giá", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.show()

def Pie_chart(House_List):
    Location_Counts = House_List['Location'].value_counts()

    plt.figure(figsize=(15, 10))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("pastel", len(Location_Counts))

    # Vẽ biểu đồ pie
    wedges, texts, autotexts = plt.pie(Location_Counts, autopct='%1.1f%%', colors=colors)
    plt.title("Số lượng bất động sản theo khu vực", fontsize=18)
    plt.axis('equal')

    plt.legend(wedges, Location_Counts.index, title="Khu vực", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    return plt.show()





