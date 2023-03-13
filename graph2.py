# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from flask import Flask, render_template
# import time

# app = Flask(__name__)

# @app.route('/piechart')
# def piechart():
#     # Load data from output.csv
#     data = pd.read_csv('C:/Users/banda/OneDrive/Desktop/hackathon/output.csv')
    
#     # Compute counts of each complaint status
#     complaint_counts = data.groupby('Complaint status:')['Complaint status:'].count()
    
#     # Create pie chart
#     plt.figure(figsize=(5,5))
#     plt.pie(complaint_counts, labels=complaint_counts.index, autopct='%1.1f%%', colors=['red', 'green'])
#     plt.title('Complaint Status')
#     plt.axis('equal')
    
#     # Save plot to a file
#     plt.savefig('static/piechart.png', bbox_inches='tight')
    
#     # Render the pie chart on a template
#     return render_template('piechart.html')

# if __name__ == '__main__':
#     while True:
#         # Update the pie chart every 5 seconds
#         time.sleep(5)
#         piechart()

# from flask import Flask, render_template
# import pandas as pd
# import seaborn as sns

# app = Flask(__name__)

# def get_data():
#     data = pd.read_csv("output.csv")
#     return data

# def piechart():
#     data = get_data()
#     complaint_counts = data.groupby('Complaint status:')['Complaint status:'].count()

#     # Set color palette
#     colors = ['red','green']

#     # Create pie chart
#     pie = sns.color_palette(colors)
#     pie = complaint_counts.plot.pie(autopct='%1.1f%%', colors=colors)

#     # Save plot to a file
#     fig = pie.get_figure()
#     fig.savefig('static/piechart.png')

# @app.route('/')
# def index():
#     piechart()
#     return render_template('piechart.html')

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template
# import pandas as pd
# import seaborn as sns
# import time
# import os
# from flask_socketio import SocketIO, emit

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)

# def get_data():
#     data = pd.read_csv("C:/Users/banda/OneDrive/Desktop/hackathon/output.csv")
#     return data

# def piechart():
#     data = get_data()
#     complaint_counts = data.groupby('Complaint status:')['Complaint status:'].count()

#     # Set color palette
#     colors = ['red','green']

#     # Create pie chart
#     pie = sns.color_palette(colors)
#     pie = complaint_counts.plot.pie(autopct='%1.1f%%', colors=colors)

#     # Save plot to a file
#     fig = pie.get_figure()
#     fig.savefig('static/piechart.png')

#     # Emit a message to update the client
#     socketio.emit('update', {'data': 'updated'})

# @app.route('/')
# def index():
#     return render_template('piechart.html')

# @socketio.on('connect')
# def test_connect():
#     print('Client connected')
#     piechart()

# @socketio.on('disconnect')
# def test_disconnect():
#     print('Client disconnected')

# if __name__ == '__main__':
#     socketio.run(app, debug=True)


from flask import Flask, render_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import threading
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

class Watcher:
    def __init__(self, path):
        self.path = 'C:/Users/banda/OneDrive/Desktop/hackathon/output.csv'
        self.observer = Observer()
        

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'modified':
            update_piechart()

def get_data():
    data = pd.read_csv("C:/Users/banda/OneDrive/Desktop/hackathon/output.csv")
    return data

def generate_piechart():
    data = get_data()
    complaint_counts = data.groupby('Complaint status:')['Complaint status:'].count()

    # Set color palette
    colors = ['red','green']

    # Create pie chart
    pie = sns.color_palette(colors)
    pie = complaint_counts.plot.pie(autopct='%1.1f%%', colors=colors)

    # Save plot to a file
    fig = pie.get_figure()
    fig.savefig('static/piechart.png')

def update_piechart():
    generate_piechart()

@app.route('/')
def index():
    return render_template('piechart.html')

def run_watcher():
    w = Watcher('.')
    w.run()

if __name__ == '__main__':
    # Generate initial pie chart
    generate_piechart()

    # Start watcher thread
    thread = threading.Thread(target=run_watcher)
    thread.daemon = True
    thread.start()

    # Start Flask app
    app.run(debug=True)
