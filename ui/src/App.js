import React, { useState } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';


const App = () => {
  
    const [isLoading, setIsloading] = useState(false);
	
  	const [content, setContent] = useState('');

	const [predictions, setPredictions] = useState(false);

	const [plotNum, setPlotNum] = useState(0);

	const [analysis, setAnalysis] = useState(false);
	    
	const handleChange = (e) => {
		const text = e.target.value;
		setContent(text);
	}

	
    const handleSubmit = async (e) => {		
		e.preventDefault();
		if (!content) {
		  	return;
		}
		
		setIsloading(true);
		const res = await fetch("/predict", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Accept": "application/json"
			},
			body: JSON.stringify(content)
		});
		const data = await res.json();						
		setPredictions(data.prediction); 
		setPlotNum(data.url);
		setIsloading(false);		     
	}

	const handleAnalysis = async (e) => {		
		e.preventDefault();
		if (!content) {
		  	return;
		}
		
		setIsloading(true);
		const res = await fetch("/analysis", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Accept": "application/json"
			},
			body: JSON.stringify(content)
		});
		const data = await res.json();						
		setAnalysis(data.sentiment);		
		setIsloading(false);		     
	}
    
    const plot_style = {
      position: 'relative',
      margin: 'auto',
	  height: '90%',
    }

    const plot_img_style = {
      position: 'relative',
      height: '90%',
    }


    
    return(
		<Container>
			<div>
				<h2 className="title">Text Predictor</h2>
			</div>
			<div className="content">
				<Form>
					<div className="form-group">
						<label htmlFor="input">Input Text :</label>
						<textarea className="form-control" rows="5" id="input"
							onChange={handleChange} value={content}/>
					</div>
					<div className="form-row">
						<div className="form-col">
							<Button 
								type="summit"
								variant="success"
								className="btn btn-primary"
								disabled={isLoading} 
								onClick={!isLoading ? handleSubmit : null}>
								{ isLoading ? 'Predicting' : 'Predict Personality' }
							</Button>		
						</div>
						<div className="form-col">
							<Button 
								type="summit"
								variant="success"
								className="btn btn-primary"
								disabled={isLoading} 
								onClick={!isLoading ? handleAnalysis : null}>
								{ isLoading ? 'Analyzing' : 'Sentiment Analysis' }
							</Button>		
						</div>
					</div>								
				</Form>
				
				{predictions === false ? null :
					(<Row className="result-container">
						<Col>
							<h5 id="result">Your Personality Traits :</h5>
							<table className="table">
								<thead>
									<tr>
										<th scope="col">Prediction</th>
										<th scope="col">Trait Catagory</th>
										<th scope="col">Trait Score</th>
										<th scope="col">Probability of Trait</th>										
									</tr>
								</thead>
								<tbody>
									<tr>
										<th scope="row">Openness</th>
										<td>{predictions.pred_cOPN}</td>
										<td>{predictions.pred_sOPN}</td>
										<td>{predictions.pred_prob_cOPN}</td>										
									</tr>
									<tr>
										<th scope="row">Consientiousness</th>
										<td>{predictions.pred_cCON}</td>
										<td>{predictions.pred_sCON}</td>
										<td>{predictions.pred_prob_cCON}</td>
									</tr>
									<tr>
										<th scope="row">Extraversion</th>
										<td>{predictions.pred_cEXT}</td>
										<td>{predictions.pred_sEXT}</td>
										<td>{predictions.pred_prob_cEXT}</td>
									</tr>
									<tr>
										<th scope="row">Agreeableness</th>
										<td>{predictions.pred_cAGR}</td>
										<td>{predictions.pred_sAGR}</td>
										<td>{predictions.pred_prob_cAGR}</td>
									</tr>
									<tr>
										<th scope="row">Neuroticism</th>
										<td>{predictions.pred_cNEU}</td>
										<td>{predictions.pred_sNEU}</td>
										<td>{predictions.pred_prob_cNEU}</td>
									</tr>
								</tbody>
							</table>							
						</Col>									
					</Row>								
					)					
            	}
				{plotNum === 0 ? null :
					(<Row className="result-container">
						<Col>
							<h5 id="result">Traits Radar Plot :</h5>
							<div style={plot_style}>
								<img src={process.env.PUBLIC_URL + `prediction${plotNum}.png`} style={plot_img_style} alt='plot'></img>
							</div>
						</Col>		
					</Row>)
				}
				{analysis === false ? null :
					(<Row className="result-container">
						<Col>
							<h5 id="result">Your Text Sentiment is :</h5>
							<div className="mr-3">
								{analysis.toString()}
							</div>
							{analysis>=0.5 ? "Positive" : "Negative"}
						</Col>						
					</Row>)	
				}			
			</div>
		</Container>
    )
}

export default App;