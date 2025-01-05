
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Fertclassifier.pkl', 'rb'))
app = Flask(__name__)


def recommendation(Temperature,Humidity,Moisture,Nitrogen,Phosphorous,Potassium,Soil_Num,Crop_Num):
    features = np.array([[Temperature,Humidity,Moisture,Nitrogen,Phosphorous,Potassium,Soil_Num,Crop_Num]])
    prediction = model.predict(features).reshape(1,-1)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictfert():
    
    details = ''
    if request.method == 'POST':
        # Get input values from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        # Call your prediction function
        prediction = str(recommendation(temperature,humidity,moisture,N,P,K,soil_type,crop_type))[3:-3]
        if prediction == 'Urea':
            desc = "Urea fertilizer is a widely used nitrogen-rich fertilizer that provides several benefits to soil health. When applied correctly, urea breaks down into ammonia and carbon dioxide, a process facilitated by soil enzymes. This breakdown releases nitrogen, an essential nutrient for plant growth, making it available to plants. Urea helps improve soil structure, promotes the growth of beneficial soil microorganisms, and enhances the soil's water-holding capacity. It also contributes to the synthesis of proteins and chlorophyll in plants, crucial for photosynthesis and overall growth. Additionally, urea is cost-effective and easy to apply, making it a popular choice among farmers. In summary, urea fertilizer enriches the soil with nitrogen, which is vital for plant growth, improves soil structure and water retention, and supports overall plant health and productivity"

            details = 'For more details: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/urea-fertilizer'
        elif prediction == 'DAP':
            desc = 'DAP (Diammonium Phosphate) fertilizer is a popular source of phosphorus and nitrogen for plants. It provides essential nutrients that help plants grow strong and healthy. Phosphorus in DAP helps with root development, flowering, and fruiting. It also plays a crucial role in energy transfer within plants. Nitrogen, on the other hand, is important for leaf growth and overall plant vigor. When applied to soil, DAP helps improve soil fertility, enhances plant growth, and increases crop yields. It is suitable for a wide range of crops and is easy to handle and apply'

            details = 'For more details: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/diammonium-phosphate'
        elif prediction == '14-35-14':
            desc = "Contains Nitrogen, Phosphorous and Potassium.Highest total nutrient content among NPK fertilizers (63%)N & P ratio same as DAP. \nIn addition,14-35-14 has extra 14% potash.High in Phosphorous content (35%). 14-35-14 is a complex fertiliser containing all major nutrients viz. Nitrogen, Phosphorous and Potassium.The only complex having highest total nutrient content among the NPK complex fertilisers. (Total nutrients are 63%). \nN&P are available in 1:2.5 ratio as in the case of DAP.It is the highest Phosphate (35%) containing complex compared to any other NPK complexes. \nEntire Nitrogen is available in Ammonical form. 29 percent out of 35% Phosphate and entire Potash is available in water-soluble form and therefore, easily available to crops. \nThe NPK ratio 1:2.5:1 is a scientific combination for basal application to all crops and all the nutrients are chemically combined and interaction is synergic.It does not contain any filler and it has 100% nutrient containing material having secondary and micro-nutrients such as Sulphur, Calcium, Magnesium and Iron. \n14-35-14 is a suitable complex for all soils since it is neutral in nature and does not leave any acidity or alkalinity in soil.It is an ideal and suitable complex for all crops for basal application. \n14-35-14 is an ideal complex particularly for Rice, Cotton, groundnut, chillies, Soya bean, Potato and other commercial crops which require high Phosphate initially. \nHowever for chlorine sensitive crops like tobacco and grapes, application of 14-35-14 is not advisable."

            details = "For more details: https://www.indiamart.com/proddetail/gromor-14-35-14-10329096730.html"
        elif prediction == '28-28':
            desc = '100% water soluble fertiliser containing high N and P. It provides the immediate nutrition to the crop during the peak growth period. It is virtually free from detrimental elements like Chloride and Sodium. It can be spray on all types of vegetables, fruit crops, cereals and pulses for boosting the crop growth thus getting better yield and quality.Total Nitrogen: 28%,Ammonical Nitrogen: 6%, Urea Nitrogen: 22%, Water Soluble P2O5 (Cloride free foliar grade): 28%.'

            details = 'For more details: https://rythuagro.in/product/insta-npk-28280/'
        elif prediction == '17-17-17':
            desc = '''17-17-17 fertilizer contains 17% nitrogen (N), 17% phosphorus (P2O5), and 17% potassium (K2O) by weight. In addition to these primary nutrients, it typically contains secondary and micronutrients essential for plant growth, such as calcium (Ca), magnesium (Mg), sulfur (S), iron (Fe), manganese (Mn), zinc (Zn), copper (Cu), boron (B), and molybdenum (Mo). 
            
            Nitrogen facilitates the development of lush foliage and greenery, promoting photosynthesis and overall plant growth. Phosphorus supports root development, flowering, and fruiting, aiding in energy transfer within the plant. Potassium contributes to various physiological processes, including water regulation, disease resistance, and overall plant vigor. The secondary and micronutrients ensure comprehensive plant nutrition, preventing deficiencies and promoting optimal growth and productivity.'''

            details = 'For more details: https://simpleshowing.ghost.io/when-to-use-17-17-17-fertilizer-how-to-use-triple-17/'
        elif prediction == '20-20':
            desc = 'Since it is so highly concentrated, NPK 20 20 20 has specific uses and is not an everyday plant fertiliser. For example, NPK 20 20 20 is excellent for hanging baskets due to the increased nutrient leaching that occurs.  It is also appropriate to use NPK 20 20 20 in poor-quality soil. The strong fertiliser can quickly and effectively raise the nutrient content of the soil providing a suitable foundation for plants. However, it should be used for just a short period of time and then either be diluted or substituted for another, less concentrated fertiliser. '
        
            details = 'For more details: https://www.bacfertilizers.com/mineral-fertilizer/npk-20-20-20'
        elif prediction == '10-26-26':
            desc = '10-26-26 fertilizer is a type of fertilizer that contains high levels of phosphorus (P) and potassium (K), with a balanced amount of nitrogen (N). This fertilizer provides essential nutrients to the soil, promoting strong root development, flowering, and fruiting in plants. Phosphorus is crucial for energy transfer and root growth, while potassium helps with overall plant vigor and disease resistance. The balanced N-P-K ratio in 10-26-26 fertilizer makes it suitable for a wide range of crops, helping improve soil fertility and increasing crop yields. It is particularly beneficial for flowering and fruiting plants.'

            details = 'For more details: https://mahadhan.co.in/product-portfolio/enhanced-efficiency-fertilizers/smartek/mahadhan-smartek-102626'
        else:
            desc = "No recommendation found."

            details = ''

        return render_template('recommendation.html', fertilizer=prediction, description=desc, details = details)
    

if __name__ == '__main__':
    app.run(debug=True)
