from flask import Flask, send_file
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class StockData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(120), nullable=False)
    trading_code = db.Column(db.String(120), nullable=False)
    opening_price = db.Column(db.Float, nullable=False)
    closing_price = db.Column(db.Float)
    yesterdays_closing_price = db.Column(db.Float)
    trade = db.Column(db.Integer)
    value_mn = db.Column(db.Float)
    volume = db.Column(db.Integer)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/import-data')
def import_data():
    json_files = [
        'C:/Users/User/Documents/DSEX/data/prices_2008.json',
        'C:/Users/User/Documents/DSEX/data/prices_2009.json',
        'C:/Users/User/Documents/DSEX/data/prices_2010.json',
        'C:/Users/User/Documents/DSEX/data/prices_2011.json',
        'C:/Users/User/Documents/DSEX/data/prices_2012.json',
        'C:/Users/User/Documents/DSEX/data/prices_2013.json',
        'C:/Users/User/Documents/DSEX/data/prices_2014.json',
        'C:/Users/User/Documents/DSEX/data/prices_2015.json',
        'C:/Users/User/Documents/DSEX/data/prices_2016.json',
        'C:/Users/User/Documents/DSEX/data/prices_2017.json',
        'C:/Users/User/Documents/DSEX/data/prices_2018.json',
        'C:/Users/User/Documents/DSEX/data/prices_2019.json',
        'C:/Users/User/Documents/DSEX/data/prices_2020.json',
        'C:/Users/User/Documents/DSEX/data/prices_2021.json',
        'C:/Users/User/Documents/DSEX/data/prices_2022.json',

        
    ]
    
    for file_path in json_files:
        df = pd.read_json(file_path)
        for _, row in df.iterrows():
            stock_data = StockData(
                date=row['date'],
                trading_code=row['trading_code'],
                opening_price=row['opening_price'],
                closing_price=row.get('closing_price'),
                yesterdays_closing_price=row.get('yesterdays_closing_price'),
                trade=row.get('trade'),
                value_mn=row.get('value_mn'),
                volume=row.get('volume'),
            )
            db.session.add(stock_data)
        db.session.commit()
    
    return "Data imported successfully!"

@app.route('/graph/historical/<trading_code>')
def historical_graph(trading_code):
    data = StockData.query.filter_by(trading_code=trading_code).all()
    df = pd.DataFrame([(d.date, d.opening_price) for d in data], columns=['Date', 'Opening Price'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Opening Price'], marker='', color='blue', linewidth=2, label=trading_code)
    plt.title(f'Historical Opening Prices for {trading_code}')
    plt.xlabel('Date')
    plt.ylabel('Opening Price')
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)