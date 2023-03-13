import pandas as pd
import seaborn as sns

# # Load data from the CSV file
# df = pd.read_csv("C:/Users/banda/OneDrive/Desktop/hackathon/output.csv")

# # Create a count plot to show the number of complaints and non-complaints
# sns.countplot(x="Complaint status:", data=df)

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('C:/Users/banda/OneDrive/Desktop/hackathon/output.csv')

# Group the data by complaint status and count the number of occurrences
complaint_counts = data.groupby('Complaint status:')['Complaint status:'].count()

# Plot the data as a bar chart
# complaint_counts.plot(kind='bar')
# plt.title('Complaints vs Non-Complaints')
# plt.xlabel('Complaint Status')
# plt.ylabel('Number of Occurrences')
# plt.show()

# Plot the data as a pie chart
labels = ['Complaint', 'Not a Complaint']
sizes = [complaint_counts['Complaint'], complaint_counts['Not a Complaint']]
colors = ['red', 'green']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()


# plt.bar(complaint_counts.index, complaint_counts.values)
# plt.xlabel('Complaint Status')
# plt.ylabel('Number of Complaints')
# plt.title('Number of Complaints and Not-Complaints Received by the Bank')
# plt.show()

# # Set the color palette
# colors = ['red', 'green']
# sns.set_palette(sns.color_palette(colors))
# # sns.set_palette('pastel')
# # create countplot
# ax = sns.countplot(x='Complaint status:', data=data)

# # set y-axis labels
# counts = data['Complaint status:'].value_counts()
# ax.set_yticklabels(counts.values)

# # set title
# ax.set_title('Complaints vs Not Complaints')

# # show plot
# plt.show()
# # Create a countplot of the complaint status
# sns.countplot(x='Complaint status:', data=data)

# # Set the title and axis labels
# plt.title('Number of Complaints vs. Non-Complaints')
# plt.xlabel('Complaint Status')
# plt.ylabel('Count')

# # Show the plot
# plt.show()
