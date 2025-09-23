from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mediapipe as mp
import datetime
from typing import Dict, List, Tuple
import base64
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class FacialFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_features(self, image_np: np.ndarray) -> Dict:
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in image")
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image_np.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        features = self._calculate_facial_ratios(points, w, h)
        features.update(self._extract_geometric_features(points))
        features.update(self._analyze_facial_symmetry(points))
        features.update(self._calculate_facial_angles(points))
        return features
    
    def _calculate_facial_ratios(self, points: List[Tuple], w: int, h: int) -> Dict:
        left_eye_left = points[33]
        left_eye_right = points[133]
        right_eye_left = points[362]
        right_eye_right = points[263]

        nose_tip = points[1]
        nose_bridge = points[9]

        mouth_left = points[61]
        mouth_right = points[291]
        mouth_top = points[13]
        mouth_bottom = points[14]

        chin = points[175]
        forehead = points[10]

        left_cheek = points[116]
        right_cheek = points[345]

        face_width = abs(left_cheek[0] - right_cheek[0])
        face_height = abs(forehead[1] - chin[1])
        eye_distance = abs(left_eye_right[0] - right_eye_left[0])
        eye_width_left = abs(left_eye_left[0] - left_eye_right[0])
        eye_width_right = abs(right_eye_left[0] - right_eye_right[0])
        nose_width = abs(mouth_left[0] - mouth_right[0]) * 0.6
        mouth_width = abs(mouth_left[0] - mouth_right[0])
        mouth_height = abs(mouth_top[1] - mouth_bottom[1])
        nose_length = abs(nose_tip[1] - nose_bridge[1])
        forehead_height = abs(forehead[1] - left_eye_left[1])

        return {
            'face_width': face_width,
            'face_height': face_height,
            'face_ratio': face_height / face_width if face_width > 0 else 1,
            'eye_distance': eye_distance,
            'eye_distance_ratio': eye_distance / face_width if face_width > 0 else 0,
            'eye_width_avg': (eye_width_left + eye_width_right) / 2,
            'eye_width_ratio': ((eye_width_left + eye_width_right) / 2) / face_width if face_width > 0 else 0,
            'nose_width': nose_width,
            'nose_width_ratio': nose_width / face_width if face_width > 0 else 0,
            'mouth_width': mouth_width,
            'mouth_width_ratio': mouth_width / face_width if face_width > 0 else 0,
            'mouth_height': mouth_height,
            'nose_length': nose_length,
            'nose_length_ratio': nose_length / face_height if face_height > 0 else 0,
            'forehead_height': forehead_height,
            'forehead_ratio': forehead_height / face_height if face_height > 0 else 0
        }
    
    def _extract_geometric_features(self, points: List[Tuple]) -> Dict:
        jaw_points = [points[i] for i in [172, 136, 150, 149, 176, 148, 152, 377, 400,
                                378, 379, 365, 397, 288, 361, 323]]
        jaw_curvature = self._calculate_curvature(jaw_points)

        contour_points = [points[i] for i in [10, 151, 9, 8, 168, 6, 148,176, 149,
                                  150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]]
        face_roundness = self._calculate_roundness(contour_points)

        return {
            'jaw_curvature': jaw_curvature,
            'face_roundness': face_roundness,
        }
        
    def _analyze_facial_symmetry(self, points: List[Tuple]) -> Dict:
        nose_bridge = points[9]
        center_x = nose_bridge[0]

        left_eye = points[33]
        right_eye = points[263]
        eye_symmetry = abs((center_x - left_eye[0]) - (right_eye[0] - center_x)) / abs(right_eye[0] - left_eye[0]) if abs(right_eye[0] - left_eye[0]) > 0 else 0

        mouth_left = points[61]
        mouth_right = points[291]
        mouth_symmetry = abs((center_x - mouth_left[0]) - (mouth_right[0] - center_x)) / abs(mouth_right[0] - mouth_left[0]) if abs(mouth_right[0] - mouth_left[0]) > 0 else 0

        return {
            'eye_symmetry': 1 - eye_symmetry,
            'mouth_symmetry': 1 - mouth_symmetry,
            'overall_symmetry': (2 - eye_symmetry - mouth_symmetry) / 2
        }
        
    def _calculate_facial_angles(self, points: List[Tuple]) -> Dict:
        nose_tip = points[1]
        nose_bridge = points[9]
        nose_base = points[2]

        nose_angle = self._calculate_angle(nose_bridge, nose_tip, nose_base)

        left_jaw = points[172]
        chin = points[175]
        right_jaw = points[397]

        jaw_angle = self._calculate_angle(left_jaw, chin, right_jaw)

        return {
            'nose_angle': nose_angle,
            'jaw_angle': jaw_angle,
        }
        
    def _calculate_curvature(self, points: List[Tuple]) -> float:
        if len(points) < 3:
            return 0
        curvatures = []
        for i in range(1, len(points) -1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
            dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
            cross_product = dx1 * dy2 - dy1 * dx2
            length1 = np.sqrt(dx1**2 + dy1**2)
            length2 = np.sqrt(dx2**2 + dy2**2)
            if length1 * length2 > 0:
                curvature = abs(cross_product) / (length1 * length2)
                curvatures.append(curvature)
        return np.mean(curvatures) if curvatures else 0
    
    def _calculate_roundness(self, points: List[Tuple]) -> float:
        if len(points) < 4:
            return 0
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        distances = [np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in points]
        std_dist = np.std(distances)
        mean_dist = np.mean(distances)
        return 1/(1 + std_dist/mean_dist) if mean_dist > 0 else 0
        
    def _calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)*180/np.pi

class AstrologyMLPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self._train_models()
    
    def _generate_training_data(self, n_samples: int = 5000) -> pd.DataFrame:
        np.random.seed(42)
        data = []
        for _ in range(n_samples):
            features = {
                'face_ratio': np.random.normal(1.3, 0.2),
                'eye_distance_ratio': np.random.normal(0.3, 0.05),
                'eye_width_ratio': np.random.normal(0.15, 0.03),
                'nose_width_ratio': np.random.normal(0.25, 0.05),
                'mouth_width_ratio': np.random.normal(0.4, 0.08),
                'nose_length_ratio': np.random.normal(0.15, 0.03),
                'forehead_ratio': np.random.normal(0.25, 0.05),
                'jaw_curvature': np.random.normal(0.5, 0.15),
                'face_roundness': np.random.normal(0.6, 0.2),
                'eye_symmetry': np.random.normal(0.8, 0.1),
                'mouth_symmetry': np.random.normal(0.8, 0.1),
                'overall_symmetry': np.random.normal(0.8, 0.1),
                'nose_angle': np.random.normal(90, 15),
                'jaw_angle': np.random.normal(120, 20)
            }
            college_backs = max(0, int(
                5*(1-features['overall_symmetry']) +
                3*(1-features['forehead_ratio']) +
                np.random.normal(2,1.5)
            ))
            marriage_age = int(
                22 + 8*features['face_roundness'] +
                5*(1-features['eye_symmetry']) +
                np.random.normal(3,2)
            )
            marriage_age = max(18,min(40,marriage_age))
            career_success = min(10,max(1,
                5 + 3*features['jaw_curvature'] +
                2*features['overall_symmetry'] +
                np.random.normal(0,1)
            ))
            wealth_level = min(10,max(1,
                4 + 4*features['nose_length_ratio']*10 +
                2*features['mouth_width_ratio']*5 +
                np.random.normal(1,1.5)
            ))
            health_score = min(10,max(1,
                6 + 2*features['overall_symmetry'] +
                2*features['face_roundness'] +
                np.random.normal(0,1)
            ))
            record = {**features}
            record.update({
                'college_backs': college_backs,
                'marriage_age': marriage_age,
                'career_success': career_success,
                'wealth_level': wealth_level,
                'health_score': health_score
            })
            data.append(record)
        return pd.DataFrame(data)
    
    def _train_models(self):
        print("Generating training data and training models...")
        df = self._generate_training_data()
        feature_cols = [
            'face_ratio', 'eye_distance_ratio', 'eye_width_ratio',
            'nose_width_ratio', 'mouth_width_ratio', 'nose_length_ratio',
            'forehead_ratio', 'jaw_curvature', 'face_roundness',
            'eye_symmetry', 'mouth_symmetry', 'overall_symmetry',
            'nose_angle', 'jaw_angle'
        ]
        target_cols = ['college_backs', 'marriage_age', 'career_success', 'wealth_level', 'health_score']
        X = df[feature_cols]

        for target in target_cols:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            self.models[target] = model
            self.scalers[target] = scaler

            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            print(f"{target} - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")

    def predict_life_events(self, facial_features: Dict) -> Dict:
        feature_order = [
            'face_ratio', 'eye_distance_ratio', 'eye_width_ratio',
            'nose_width_ratio', 'mouth_width_ratio', 'nose_length_ratio',
            'forehead_ratio', 'jaw_curvature', 'face_roundness',
            'eye_symmetry', 'mouth_symmetry', 'overall_symmetry',
            'nose_angle', 'jaw_angle'
        ]
        for feature in feature_order:
            if feature not in facial_features:
                facial_features[feature] = 0.5

        feature_vector = np.array([[facial_features[f] for f in feature_order]])
        predictions = {}

        for target, model in self.models.items():
            scaler = self.scalers[target]
            scaled_features = scaler.transform(feature_vector)
            prediction = model.predict(scaled_features)[0]

            if target == 'college_backs':
                predictions[target] = max(0, int(round(prediction)))
            elif target == 'marriage_age':
                predictions[target] = max(18, min(40, int(round(prediction))))
            else:
                predictions[target] = max(1, min(10, prediction))

        return predictions


class AstrologyPersonalityAnalyzer:
    def analyze_personality(self, facial_features: Dict) -> Dict:
        personality = {}
        confidence_score = (
            facial_features.get('jaw_curvature', 0.5)*0.4 +
            facial_features.get('overall_symmetry', 0.8)*0.3 +
            facial_features.get('eye_width_ratio', 0.15)*10*0.3
        )
        personality['confidence'] = min(10, max(1, confidence_score * 10))

        intelligence_score = (
            facial_features.get('forehead_ratio', 0.25)*4*0.5 +
            facial_features.get('eye_distance_ratio', 0.3)*3*0.3 +
            facial_features.get('overall_symmetry', 0.8)*0.2
        )
        personality['intelligence'] = min(10, max(1, intelligence_score * 10))

        creativity_score = (
            facial_features.get('face_roundness', 0.6)*0.4 +
            (1-facial_features.get('overall_symmetry', 0.8))*0.3 +
            facial_features.get('nose_angle', 90)/90*0.3
        )
        personality['creativity'] = min(10, max(1, creativity_score * 10))

        social_score = (
            facial_features.get('mouth_width_ratio', 0.4)*2.5*0.4 +
            facial_features.get('eye_symmetry', 0.8)*0.3 +
            facial_features.get('face_roundness', 0.6)*0.3
        )
        personality['social_skills'] = min(10, max(1, social_score * 10))

        leadership_score = (
            facial_features.get('jaw_curvature', 0.5)*0.4 +
            facial_features.get('nose_length_ratio', 0.15)*6.67*0.3 +
            facial_features.get('overall_symmetry', 0.8)*0.3
        )
        personality['leadership'] = min(10, max(1, leadership_score * 10))

        return personality


class AstrologyEngine:
    def __init__(self):
        self.feature_extractor = FacialFeatureExtractor()
        self.ml_predictor = AstrologyMLPredictor()
        self.personality_analyzer = AstrologyPersonalityAnalyzer()
        self.lucky_elements = {
            'colors': ['Red', 'Blue', 'Green', 'Gold', 'Silver', 'Purple', 'Orange', 'Pink'],
            'numbers': [1, 3, 7, 9, 11, 13, 21, 27, 33, 44, 77],
            'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'gemstones': ['Ruby', 'Sapphire', 'Emerald', 'Diamond', 'Pearl', 'Topaz', 'Amethyst', 'Opal'],
            'directions': ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
        }
        self.career_fields = [
            {
                'field': 'Technology & Engineering',
                'description': 'Software development, AI, robotics, engineering',
                'keywords': ['analytical', 'logical', 'innovative']
            },
            {
                'field': 'Business & Finance',
                'description': 'Entrepreneurship, banking, investment, consulting',
                'keywords': ['strategic', 'confident', 'leadership']
            },
            {
                'field': 'Creative Arts',
                'description': 'Design, music, writing, filmmaking, art',
                'keywords': ['creative', 'expressive', 'intuitive']
            },
            {
                'field': 'Healthcare & Medicine',
                'description': 'Doctor, nurse, therapist, medical research',
                'keywords': ['caring', 'detail-oriented', 'empathetic']
            },
            {
                'field': 'Education & Research',
                'description': 'Teaching, academic research, training',
                'keywords': ['intelligent', 'patient', 'communicative']
            },
            {
                'field': 'Social Services',
                'description': 'Social work, counseling, human resources',
                'keywords': ['empathetic', 'social', 'helpful']
            }
        ]

    def analyze_face(self, image_np: np.ndarray) -> Dict:
        try:
            facial_features = self.feature_extractor.extract_features(image_np)
            ml_predictions = self.ml_predictor.predict_life_events(facial_features)
            personality = self.personality_analyzer.analyze_personality(facial_features)
            predictions = {
                'facial_features': facial_features,
                'life_predictions': ml_predictions,
                'personality_traits': personality,
                'lucky_elements': self._generate_lucky_elements(facial_features),
                'career_guidance': self._generate_career_guidance(personality, facial_features),
                'analysis_timestamp': datetime.datetime.now().isoformat(),
            }
            return predictions
        except Exception as e:
            return {'error': str(e)}

    def _generate_lucky_elements(self, features: Dict) -> Dict:
        face_hash = hash(str(sorted(features.items())))
        np.random.seed(abs(face_hash) % 1000000)
        return {
            'color': np.random.choice(self.lucky_elements['colors']),
            'number': np.random.choice(self.lucky_elements['numbers']),
            'day': np.random.choice(self.lucky_elements['days']),
            'gemstone': np.random.choice(self.lucky_elements['gemstones']),
            'direction': np.random.choice(self.lucky_elements['directions'])
        }

    def _generate_career_guidance(self, personality: Dict, features: Dict) -> Dict:
        career_scores = []

        for career in self.career_fields:
            score = 0
            if career['field'] == 'Technology & Engineering':
                score = personality.get('intelligence', 5)*0.4 + personality.get('creativity', 5)*0.3 + (10 - personality.get('social_skills', 5))*0.3
            elif career['field'] == 'Business & Finance':
                score = personality.get('confidence', 5)*0.4 + personality.get('leadership', 5)*0.4 + personality.get('intelligence', 5)*0.2
            elif career['field'] == 'Creative Arts':
                score = personality.get('creativity', 5)*0.5 + personality.get('social_skills', 5)*0.3 + (10 - personality.get('confidence', 5))*0.2
            elif career['field'] == 'Healthcare & Medicine':
                score = personality.get('intelligence', 5)*0.3 + personality.get('social_skills', 5)*0.4 + (features.get('overall_symmetry', 0.8)*10)*0.3
            elif career['field'] == 'Education & Research':
                score = personality.get('intelligence', 5)*0.4 + personality.get('social_skills', 5)*0.3 + (features.get('forehead_ratio', 0.25)*40)*0.3
            elif career['field'] == 'Social Services':
                score = personality.get('social_skills', 5)*0.5 + (10 - personality.get('confidence', 5))*0.3 + personality.get('creativity', 5)*0.2
            career_scores.append((career, score))

        career_scores.sort(key=lambda x: x[1], reverse=True)
        top_careers = career_scores[:3]

        return {
            'primary_recommendation': {
                'field': top_careers[0][0]['field'],
                'description': top_careers[0][0]['description'],
                'compatibility_score': round(top_careers[0][1], 1),
                'reasoning': 'Your personality profile strongly aligns with this career path, based on your traits and facial features.'
            },
            'secondary_recommendations': [
                {
                    'field': career[0]['field'],
                    'description': career[0]['description'],
                    'compatibility_score': round(career[1], 1)
                } for career in top_careers[1:]
            ]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data_url = request.json['image']
        img_str = re.search(r'base64,(.*)', data_url).group(1)
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = astrology_engine.analyze_face(img_np)
        safe_result = convert_numpy_types(result)
        return jsonify(safe_result)
    except Exception as e:
        print("Error during /analyze:", e)
        return jsonify({'error': str(e)})

astrology_engine = AstrologyEngine()

if __name__ == "__main__":
    app.run(debug=True)
