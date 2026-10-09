import hashlib
import io
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support
from core import analyze, export_csv, export_report, record_context
from demo import generate_demo
from ollama_client import explain, list_models

st.set_page_config(page_title='Üretim Anomali Dedektörü', page_icon='🏭', layout='wide')
st.title('Üretim Anomali Dedektörü')
st.caption('Python ile davranış farklılıklarını bul, yerel Ollama ile gözlemleri açıkla.')
st.info('Bu bir öğrenme projesidir. İşaretlenen kayıt farklı bir davranış gösterir; kesin arıza teşhisi değildir.')

with st.sidebar:
    st.header('Veri ve model')
    source = st.radio('Veri kaynağı', ['Sentetik demo', 'CSV yükle'])
    upload = st.file_uploader('Üretim kayıtları', type=['csv']) if source == 'CSV yükle' else None
    sep = st.selectbox('CSV ayırıcı', [',', ';']) if source == 'CSV yükle' else ','
    ratio = st.slider('Eğitim için geçmiş veri oranı', .5, .85, .7, .05)
    contamination = st.slider('Eğitimde varsayılan anomali oranı', .01, .25, .08, .01)
    st.caption('Bu oran eğitim skorlarının karar eşiğini belirler. Testte aynı oranda anomali çıkması gerekmez.')
    machine_specific = st.checkbox('Makineye özel model', value=False,
        help='Her makine kendi geçmişinden öğrenir. Eğitimde 30 kaydı olmayan makineler ortak modeli kullanır.')
    train_clicked = st.button('Modeli eğit ve analiz et', type='primary', width='stretch')

try:
    if source == 'CSV yükle':
        if upload is None:
            st.write('CSV seçin. Gerekli kolonlar aşağıdaki veri sözlüğünde yer alıyor.')
            st.code('timestamp,machine,production_count,scrap_count,downtime_minutes,shift_minutes')
            st.stop()
        raw = upload.getvalue()
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError('CSV en fazla 10 MB olabilir.')
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding='utf-8-sig')
        fingerprint = hashlib.sha256(raw).hexdigest()
    else:
        df = generate_demo()
        fingerprint = 'demo-v1'
        st.caption('720 sentetik kayıt · 4 makine · 180 gün. Etiketler yalnızca ölçüm için kullanılır.')
    model_scope = 'per_machine' if machine_specific else 'shared'
    config = (source, fingerprint, sep, ratio, contamination, model_scope)
    if st.session_state.get('analysis_config') != config:
        st.session_state.pop('analysis', None)
        st.session_state.pop('explanations', None)
    if train_clicked:
        with st.spinner('Geçmiş kayıtlardan model öğreniliyor…'):
            st.session_state['analysis'] = analyze(df, contamination, ratio, model_scope)
            st.session_state['analysis_config'] = config
    if 'analysis' not in st.session_state:
        st.subheader('Veri önizlemesi')
        st.dataframe(df.head(20), hide_index=True, width='stretch')
        st.write('Sol menüden **Modeli eğit ve analiz et** düğmesine basın.')
        st.stop()
except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
    st.error(str(exc))
    st.stop()

result = st.session_state['analysis']
a, b, c, d = st.columns(4)
a.metric('Eğitim kaydı', len(result.train))
b.metric('Test kaydı', len(result.test))
c.metric('Şüpheli test kaydı', int(result.test['predicted_anomaly'].sum()))
d.metric('Testte işaretlenen oran', f"{result.test['predicted_anomaly'].mean():.1%}")
st.caption(f"Eğitim sonu: {result.train.timestamp.max().date()} · Test başlangıcı: {result.test.timestamp.min().date()}")
if result.unseen_machines:
    st.warning('Eğitimde görülmeyen makineler: ' + ', '.join(result.unseen_machines) + '. Sonuçları dikkatle inceleyin.')
if result.fallback_machines:
    st.warning('Makineye özel eğitim için yeterli geçmişi olmayan makinelerde ortak model kullanıldı: '
               + ', '.join(result.fallback_machines) + '.')

overview, inspect, learn = st.tabs(['Sonuçlar', 'Kayıt incele', 'Nasıl öğreniyor?'])
with overview:
    selected_machines = st.multiselect('Makine filtresi', sorted(result.test.machine.unique()))
    min_date = result.test['timestamp'].dt.date.min()
    max_date = result.test['timestamp'].dt.date.max()
    selected_dates = st.date_input('Tarih aralığı', value=(min_date, max_date), min_value=min_date, max_value=max_date)
    filtered = result.test
    if selected_machines:
        filtered = filtered[filtered.machine.isin(selected_machines)]
    if len(selected_dates) == 2:
        filtered = filtered[filtered['timestamp'].dt.date.between(selected_dates[0], selected_dates[1])]
    st.subheader('Makine bazında izleme')
    machine_summary = filtered.groupby('machine').agg(
        test_records=('machine', 'size'),
        anomaly_records=('predicted_anomaly', 'sum'),
        anomaly_rate=('predicted_anomaly', 'mean'),
        mean_anomaly_score=('anomaly_score', 'mean'),
        max_anomaly_score=('anomaly_score', 'max'),
    ).reset_index().rename(columns={
        'machine': 'Makine', 'test_records': 'Test kaydı', 'anomaly_records': 'İşaretlenen kayıt',
        'anomaly_rate': 'İşaretlenme oranı', 'mean_anomaly_score': 'Ortalama skor',
        'max_anomaly_score': 'En yüksek skor',
    })
    st.dataframe(machine_summary.sort_values('İşaretlenme oranı', ascending=False),
                 hide_index=True, width='stretch', column_config={
                     'İşaretlenme oranı': st.column_config.NumberColumn(format='%.1%'),
                     'Ortalama skor': st.column_config.NumberColumn(format='%.3f'),
                     'En yüksek skor': st.column_config.NumberColumn(format='%.3f'),
                 })
    st.caption('İşaretlenme oranı, testte anomali olarak işaretlenen kayıt payıdır; arıza olasılığı değildir.')
    daily = (filtered.assign(day=filtered['timestamp'].dt.floor('D'))
             .groupby('day').agg(anomaly_score=('anomaly_score', 'max'),
                                 anomaly_count=('predicted_anomaly', 'sum')).reset_index())
    fig = px.line(daily, x='day', y='anomaly_score', title='Her günün en yüksek anomali skoru')
    fig.add_hline(y=0, line_dash='dash', annotation_text='Karar eşiği')
    st.plotly_chart(fig, width='stretch')
    st.caption('Skor > 0 ise kayıt işaretlenir. Skor bir olasılık veya yüzde güven değildir.')
    count_fig = px.bar(daily, x='day', y='anomaly_count', title='Günlük işaretlenen kayıt sayısı')
    st.plotly_chart(count_fig, width='stretch')
    st.caption('Sayı, seçili makine ve tarih filtrelerindeki işaretli kayıtları gösterir; arıza sayısı değildir.')
    only_anomalies = st.checkbox('Yalnızca işaretlenen kayıtlar', value=True)
    shown = filtered
    if only_anomalies:
        shown = shown[shown.predicted_anomaly == 1]
    st.dataframe(shown.sort_values('anomaly_score', ascending=False), hide_index=True, width='stretch')
    st.download_button('Görünen kayıtları CSV indir', export_csv(shown), 'anomaly_results.csv', 'text/csv')
    report_metadata = {
        'source': source,
        'dataset_fingerprint': fingerprint,
        'training_ratio': ratio,
        'contamination': contamination,
        'model_scope': model_scope,
    }
    st.download_button('Tam analiz raporunu ZIP indir', export_report(result, report_metadata),
                       'anomaly_report.zip', 'application/zip')
    if result.metrics:
        st.subheader('Test etiketlerine göre değerlendirme')
        if source == 'Sentetik demo':
            st.caption('Bu skorlar sentetik anomalileri bulma başarısıdır; gerçek fabrika başarısını göstermez.')
        m = result.metrics
        if 'by_machine' not in m:
            m['by_machine'] = {}
            for machine, group in result.test.groupby('machine', sort=True):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    group['is_anomaly'], group['predicted_anomaly'], average='binary', zero_division=0)
                m['by_machine'][str(machine)] = {
                    'records': len(group), 'positive_labels': int(group['is_anomaly'].sum()),
                    'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                }
        p, r, f = st.columns(3)
        p.metric('Precision', f"{m['precision']:.3f}")
        r.metric('Recall', f"{m['recall']:.3f}")
        f.metric('F1', f"{m['f1']:.3f}")
        st.dataframe(pd.DataFrame(m['confusion_matrix'], index=['Gerçek normal', 'Gerçek anomali'],
                                 columns=['Tahmin normal', 'Tahmin anomali']), width='stretch')
        machine_metrics = pd.DataFrame(m['by_machine']).T.rename_axis('Makine').reset_index()
        machine_metrics = machine_metrics.rename(columns={
            'records': 'Test kaydı', 'positive_labels': 'Etiketli anomali',
            'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})
        st.subheader('Makine bazında performans')
        st.caption('Skorlar yalnızca kronolojik test döneminden hesaplanır; az sayıda etiketli anomali sonuçları oynaklaştırabilir.')
        st.dataframe(machine_metrics.sort_values('F1'), hide_index=True, width='stretch')
        if not m['positive_labels'] or not m['negative_labels']:
            st.warning('Test verisinde iki sınıf birlikte yok; değerlendirme sınırlı.')
        st.download_button('Ölçümleri JSON indir', json.dumps(m, ensure_ascii=False, indent=2), 'metrics.json', 'application/json')
    else:
        st.info('is_anomaly etiketi bulunmadığı için precision, recall ve F1 hesaplanmıyor.')

with inspect:
    indices = result.test.sort_values('anomaly_score', ascending=False).index.tolist()
    idx = st.selectbox('İncelenecek test kaydı', indices,
        format_func=lambda i: f"{result.test.loc[i, 'timestamp'].date()} · {result.test.loc[i, 'machine']} · skor {result.test.loc[i, 'anomaly_score']:.3f}")
    ctx = record_context(result, idx)
    feature_details = pd.DataFrame(ctx['features']).T.rename(columns={
        'value': 'Kayıt değeri', 'training_median': 'Geçmiş medyan',
        'training_p10': 'Geçmiş P10', 'training_p90': 'Geçmiş P90',
        'range_status': 'Geçmiş aralığı'})
    feature_details['Geçmiş aralığı'] = feature_details['Geçmiş aralığı'].map({
        'below': 'Altında', 'within': 'İçinde', 'above': 'Üstünde'})
    st.dataframe(feature_details, width='stretch')
    st.caption('Referans: eğitimdeki aynı makine kayıtları; bulunamazsa tüm eğitim kayıtları. Bunlar model özellik katkıları değildir.')
    model = st.text_input('Yüklü Ollama model adı', placeholder='ollama list çıktısındaki ad')
    if st.button('Yüklü modelleri göster'):
        try:
            st.write(list_models())
        except Exception:
            st.warning('Yerel Ollama erişilemiyor. Ollama uygulamasını açıp ollama list ile kontrol edin.')
    st.caption('Açıklama isteğinde yalnızca seçilen kayıt ve geçmiş istatistikleri yerel Ollama’ya gönderilir.')
    key = (idx, model)
    if st.button('Ollama ile açıkla', disabled=not model.strip()):
        try:
            with st.spinner('Yerel model açıklama hazırlıyor…'):
                st.session_state.setdefault('explanations', {})[key] = explain(ctx, model)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    answer = st.session_state.get('explanations', {}).get(key)
    if answer:
        st.write(answer['summary'])
        st.markdown('**Gözlemler**')
        for text in answer['observations']:
            st.write('• ' + text)
        st.markdown('**Önerilen kontroller**')
        for text in answer['checks']:
            st.write('• ' + text)
        st.warning(answer['limitation'])

with learn:
    st.markdown('''### Model ve Ollama iki ayrı iş yapar
1. CSV doğrulanır; hatalı adetler ve süreler reddedilir.
2. Hata oranı, duruş oranı ve çalışan dakika başına üretim hesaplanır.
3. Zaman noktalarının ilk bölümü eğitim, sonraki bölümü test olur. Aynı zaman iki bölüme dağılmaz.
4. Isolation Forest yalnızca eğitimdeki dört sayısal özelliği öğrenir. Etiket, tarih ve makine adı kullanılmaz.
5. Test kayıtları öğrenilen model ve eşikle değerlendirilir.
6. Ollama seçilen kaydın değerlerini geçmiş istatistiklerle yorumlar; anomali kararını değiştirmez.

**Precision:** İşaretlenen kayıtların ne kadarı gerçekten etiketli anomali?

**Recall:** Etiketli anomalilerin ne kadarı yakalandı?

**F1:** Precision ve recall'un harmonik ortalaması.

Bu ilk sürüm tüm makineler için ortak model kullanır. Ürün ve makine hızları çok farklıysa normal farklar
anomali sayılabilir. Sonraki adım ürün bağlamı ve makine başına model karşılaştırmasıdır.
Test skoruna bakarak eşiği tekrar tekrar seçersen test verisine uyum sağlamış olursun;
ciddi karşılaştırmalarda ayrı doğrulama dönemi ve en son dokunulmamış test dönemi kullan.''')
