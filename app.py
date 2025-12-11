# Sistema completo web con dashboard, API REST, autenticación Google, predicción y reportes
# -----------------------------------------------------------------------------
# Este archivo es un scaffolding (app.py) para una aplicación Flask que incluye:
# - Autenticación con Google (OAuth2) para iniciar sesión al entrar a la página
# - Control de usuarios y múltiples negocios (each user can have many businesses)
# - Registro de movimientos (ventas/gastos) con categorías
# - Inventario y ventas de inventario
# - Dashboard con gráficos (matplotlib) y reportes por categoría
# - Exportación a CSV/Excel
# - Predicción sencilla con regresión lineal (scikit-learn)
# - API REST para CRUD de movimientos y productos
# - Páginas imprimibles (tickets/facturas HTML que se pueden imprimir desde el navegador)
# - Nota: este es un scaffold. Requiere configurar credenciales de Google OAuth2 y correr en entorno con dependencias.

# Requisitos (sugeridos)
# pip install flask sqlalchemy flask_sqlalchemy authlib pandas matplotlib scikit-learn openpyxl flask_cors

from flask import Flask, redirect, url_for, session, request, render_template_string, jsonify, send_file
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta
import os
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ----------------- Configuración básica -----------------
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret_change_me')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configurar OAuth con Google: debes crear credenciales en Google Cloud
# y establecer las variables de entorno GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET
oauth = OAuth(app)
CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'

oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url=CONF_URL,
    client_kwargs={'scope': 'openid email profile'}
)

db = SQLAlchemy(app)

# ----------------- Modelos -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
    name = db.Column(db.String)
    businesses = db.relationship('Business', backref='owner', lazy=True)

class Business(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    products = db.relationship('Product', backref='business', lazy=True)
    movements = db.relationship('Movement', backref='business', lazy=True)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.Integer, db.ForeignKey('business.id'), nullable=False)
    name = db.Column(db.String, nullable=False)
    cost = db.Column(db.Float, default=0.0)
    price = db.Column(db.Float, default=0.0)
    stock = db.Column(db.Integer, default=0)

class Movement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.Integer, db.ForeignKey('business.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    tipo = db.Column(db.String, nullable=False)  # 'venta' or 'gasto'
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String)
    note = db.Column(db.String)

# ----------------- Inicializar DB -----------------
@app.before_first_request
def create_tables():
    db.create_all()

# ----------------- Autenticación (Google) -----------------
@app.route('/login')
def login():
    redirect_uri = url_for('auth_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def auth_callback():
    token = oauth.google.authorize_access_token()
    userinfo = oauth.google.parse_id_token(token)
    # userinfo contiene 'sub' (id), 'email', 'name'
    google_id = userinfo['sub']
    email = userinfo.get('email')
    name = userinfo.get('name')

    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User(google_id=google_id, email=email, name=name)
        db.session.add(user)
        db.session.commit()
    session['user_id'] = user.id
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ----------------- Helpers -----------------
def current_user():
    uid = session.get('user_id')
    if not uid:
        return None
    return User.query.get(uid)

# Decorator simple
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ----------------- Rutas principales -----------------
@app.route('/')
def index():
    user = current_user()
    if not user:
        # Al entrar, redirigir al login de Google
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = current_user()
    # Si el usuario no tiene negocios, mostrar opción para crearlo
    businesses = Business.query.filter_by(owner_id=user.id).all()
    # Mostrar datos agregados de primer negocio si existe
    business = businesses[0] if businesses else None

    # Render simple: include img charts
    html = """
    <h1>Dashboard - {{name}}</h1>
    {% if business %}
      <h2>Negocio: {{business.name}}</h2>
      <img src="/plot/sales?bid={{business.id}}" alt="ventas" />
      <img src="/plot/category?bid={{business.id}}" alt="por categoria" />
      <p><a href="/logout">Cerrar sesión</a></p>
    {% else %}
      <p>No tienes negocios. <a href="/create_business">Crear uno</a></p>
    {% endif %}
    """
    return render_template_string(html, name=user.name, business=business)

@app.route('/create_business', methods=['GET', 'POST'])
@login_required
def create_business():
    user = current_user()
    if request.method == 'POST':
        name = request.form['name']
        b = Business(name=name, owner_id=user.id)
        db.session.add(b)
        db.session.commit()
        return redirect(url_for('dashboard'))
    return render_template_string('''
        <form method="post">
            Nombre del negocio: <input name="name" />
            <button type="submit">Crear</button>
        </form>
    ''')

# ----------------- API REST (movements & products) -----------------
@app.route('/api/businesses', methods=['GET'])
@login_required
def api_businesses():
    user = current_user()
    bs = Business.query.filter_by(owner_id=user.id).all()
    return jsonify([{'id':b.id,'name':b.name} for b in bs])

@app.route('/api/<int:bid>/movements', methods=['GET','POST'])
@login_required
def api_movements(bid):
    if request.method == 'GET':
        movs = Movement.query.filter_by(business_id=bid).all()
        return jsonify([{
            'id':m.id,'date':m.date.isoformat(),'time':m.time.isoformat(), 'tipo':m.tipo,'amount':m.amount,'category':m.category,'note':m.note
        } for m in movs])
    data = request.json
    m = Movement(
        business_id=bid,
        date=datetime.fromisoformat(data['date']).date(),
        time=datetime.fromisoformat(data['time']).time(),
        tipo=data['tipo'],
        amount=float(data['amount']),
        category=data.get('category'),
        note=data.get('note')
    )
    db.session.add(m)
    db.session.commit()
    return jsonify({'status':'ok','id':m.id}), 201

@app.route('/api/<int:bid>/products', methods=['GET','POST'])
@login_required
def api_products(bid):
    if request.method == 'GET':
        ps = Product.query.filter_by(business_id=bid).all()
        return jsonify([{'id':p.id,'name':p.name,'cost':p.cost,'price':p.price,'stock':p.stock} for p in ps])
    data = request.json
    p = Product(business_id=bid, name=data['name'], cost=float(data.get('cost',0)), price=float(data.get('price',0)), stock=int(data.get('stock',0)))
    db.session.add(p)
    db.session.commit()
    return jsonify({'status':'ok','id':p.id}), 201

# ----------------- Reportes y gráficas -----------------
@app.route('/plot/sales')
@login_required
def plot_sales():
    bid = request.args.get('bid', type=int)
    # obtener últimos 30 días
    end = datetime.now().date()
    start = end - timedelta(days=29)
    dates = pd.date_range(start, end)
    totals = []
    for d in dates:
        vals = db.session.query(db.func.sum(Movement.amount)).filter(Movement.business_id==bid, Movement.date==d.date()).scalar() or 0
        totals.append(float(vals))

    fig, ax = plt.subplots()
    ax.plot(dates, totals)
    ax.set_title('Ventas (últimos 30 días)')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Total')

    buf = io.BytesIO()
    fig.autofmt_xdate()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/plot/category')
@login_required
def plot_category():
    bid = request.args.get('bid', type=int)
    # Agregar query que suma por categoría
    rows = db.session.query(Movement.category, db.func.sum(Movement.amount)).filter(Movement.business_id==bid).group_by(Movement.category).all()
    cats = [r[0] or 'sin-categoria' for r in rows]
    vals = [float(r[1]) for r in rows]

    fig, ax = plt.subplots()
    ax.pie(vals, labels=cats, autopct='%1.1f%%')
    ax.set_title('Gastos/Ventas por categoría')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# ----------------- Predicción (regresión lineal simple) -----------------
@app.route('/predict/<int:bid>')
@login_required
def predict_sales(bid):
    # Usar totales diarios históricos para entrenar
    rows = db.session.query(Movement.date, db.func.sum(Movement.amount)).filter(Movement.business_id==bid, Movement.tipo=='venta').group_by(Movement.date).all()
    if len(rows) < 5:
        return jsonify({'error':'Not enough data for prediction'}), 400
    df = pd.DataFrame(rows, columns=['date','total'])
    df['date_ord'] = pd.to_datetime(df['date']).map(lambda d: d.toordinal())
    X = df[['date_ord']].values
    y = df['total'].values
    model = LinearRegression().fit(X, y)
    # Predecir 7 días siguientes
    last = df['date'].max()
    preds = []
    for i in range(1,8):
        d = pd.to_datetime(last) + pd.Timedelta(days=i)
        pred = model.predict(np.array([[d.toordinal()]]))[0]
        preds.append({'date':d.date().isoformat(),'predicted':float(pred)})
    return jsonify(preds)

# ----------------- Impresión de tickets (HTML imprimible) -----------------
@app.route('/ticket/<int:mid>')
@login_required
def ticket(mid):
    m = Movement.query.get_or_404(mid)
    html = f"""
    <html>
    <head><meta charset="utf-8"><style>@media print {{ body {{ font-family: Arial; }} }}</style></head>
    <body>
      <h2>Ticket - {m.business.name}</h2>
      <p>Fecha: {m.date.isoformat()} {m.time.strftime('%H:%M:%S')}</p>
      <p>Tipo: {m.tipo}</p>
      <p>Monto: {m.amount}</p>
      <p>Categoria: {m.category}</p>
      <p>Nota: {m.note}</p>
      <script>window.print()</script>
    </body>
    </html>
    """
    return html

# ----------------- Exportes (CSV / Excel) -----------------
@app.route('/export/<int:bid>/csv')
@login_required
def export_csv(bid):
    movs = Movement.query.filter_by(business_id=bid).all()
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(['date','time','tipo','amount','category','note'])
    for m in movs:
        writer.writerow([m.date.isoformat(), m.time.strftime('%H:%M:%S'), m.tipo, m.amount, m.category, m.note])
    si.seek(0)
    return send_file(io.BytesIO(si.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='report.csv')

@app.route('/export/<int:bid>/xlsx')
@login_required
def export_xlsx(bid):
    movs = Movement.query.filter_by(business_id=bid).all()
    df = pd.DataFrame([{'date':m.date.isoformat(),'time':m.time.strftime('%H:%M:%S'),'tipo':m.tipo,'amount':m.amount,'category':m.category,'note':m.note} for m in movs])
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='movements')
    out.seek(0)
    return send_file(out, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='report.xlsx')

# ----------------- Reporte fiscal automático (simplificado) -----------------
@app.route('/fiscal/<int:bid>/<int:year>/<int:month>')
@login_required
def fiscal_report(bid, year, month):
    start = datetime(year, month, 1).date()
    if month == 12:
        end = datetime(year+1, 1, 1).date() - timedelta(days=1)
    else:
        end = datetime(year, month+1, 1).date() - timedelta(days=1)
    rows = db.session.query(Movement).filter(Movement.business_id==bid, Movement.date>=start, Movement.date<=end).all()
    total_sales = sum(r.amount for r in rows if r.tipo=='venta')
    total_expenses = sum(r.amount for r in rows if r.tipo=='gasto')
    # Este reporte es simplificado: en la práctica debes ajustar según legislación fiscal
    return jsonify({'year':year,'month':month,'sales':total_sales,'expenses':total_expenses,'profit': total_sales-total_expenses})

# ----------------- Ejecutar -----------------
if __name__ == '__main__':
    # Antes de ejecutar, exporta GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET en tu entorno
    print('Inicia la app con: flask run  (o python app.py)')
    app.run(debug=True, host='0.0.0.0', port=5000)