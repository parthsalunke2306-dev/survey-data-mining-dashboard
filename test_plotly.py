import urllib.request, json
res = urllib.request.urlopen('http://127.0.0.1:8000/api/dashboard-data').read()
data = json.loads(res)
chart_json = json.dumps(data['fdi_chart'])
html = f'''<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
</head>
<body>
    <div id="chart" style="width:600px; height:400px;"></div>
    <script>
        const chartData = {chart_json};
        Plotly.newPlot('chart', chartData.data, chartData.layout).catch(e => console.error(e));
    </script>
</body>
</html>'''
with open('test_plotly.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Saved test_plotly.html')
