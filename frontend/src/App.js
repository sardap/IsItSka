import React from 'react';
import config from './config'
import SearchForm from './coms/SearchForm'
import SearchResult from './coms/SearchResult'
import './style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class App extends React.Component {
	constructor(props) {
		super(props);

		this.search_endpoint =  window.location.origin + "/api/ska_prob";
		this.version = config.VERSION;


		const url_params = new URLSearchParams(window.location.search);

		let artist_name = url_params.get('artist_name');
		if(artist_name !== null) {
			this.artist_name = artist_name;
		}

		const track_name = url_params.get('track_name');
		if(track_name !== undefined && track_name !== null){
			this.track_name = track_name;
		}

		const track_id = url_params.get('track_id');
		if(track_id !== null) {
			this.track_id = track_id;
		}

		if(track_name == null && track_id == null) {
			this.state = {
				loading : false
			};
		} else {
			this.state = {
				loading : true
			};
		}

		this.search(track_name, artist_name, track_id);

	}

	componentDidMount() {
		document.title = "Is it Ska?"
	}

	redirectToTrack = (track_text, artist_text, track_id) => {
		console.log("redirect to track " + track_text + " " + artist_text + " " + track_id)
		let url = window.location.origin;

		if (track_text !== undefined && track_text !== null && track_text.length > 0) {
			url += "?track_name=" + track_text;
			if(artist_text !== undefined && artist_text !== null && artist_text.length > 0) {
				url += "&artist_name=" + artist_text;
			}
		} else if (track_id !== undefined && track_id !== null && track_id.length > 0) {
			url += "?track_id=" + track_id;
		} else {
			return;
		}

		window.location.href = url
	}

	search  = async (track_text, artist_text, track_id) => {
		console.log("making request track=" + track_text + " artist " + artist_text + " track_id " + track_id);

		let url = this.search_endpoint;

		if (track_text !== undefined && track_text !== null && track_text.length > 0) {
			url += "?track_name=" + track_text;
			if(artist_text !== undefined && artist_text !== null && artist_text.length > 0) {
				url += "&artist_name=" + artist_text;
			}
		} else if (track_id !== undefined && track_id !== null && track_id.length > 0) {
			url += "?track_id=" + track_id;
		} else {
			return;
		}

		this.setState({
			loading : true,
			result: undefined,
			error: undefined
		});

		let response = await fetch(url);
		console.log("Making request " + url);
		if(response.status == 200) {
			let data = await response.json();
			this.setState({
				result: data
			})
		} else if(response.status == 404) {
			this.setState({
				error : "Track does not exist"
			})			
		} else {
			console.log("Error: " + JSON.stringify(response));
		}

		this.setState({
			loading: false
		});
	}

	loading() {
		return (
			<div className="container h-100 justify-content-center vertical-center">
				<div className="container text-center">
					<div
						className="spinner-border"
						role="status"
						style={{width: "7rem", height: "7rem"}} 
					>
						<span class="sr-only">Loading...</span>
					</div>
					<div className="h1">Picking it up</div>
				</div>
			</div>
		);
	}

	content() {
		const search_element = 
			<div className="">
				<SearchForm 
					search_callback={this.redirectToTrack}
					track_name={this.track_name}
					artist_name={this.artist_name}
					track_id={this.track_id}
				/>
			</div>

		let error_info = <></>;

		if(this.state.error !== undefined) {
			error_info = 
				<div className="text-center text-danger h3 bg-white">
					{this.state.error}
				</div>
		}

		let result_info = <></>;
		
		if(this.state.result !== undefined){
			result_info = <SearchResult
				track_id = {this.state.result.track_id}
				track_link = {this.state.result.track_link}
				title = {this.state.result.title}
				artists = {this.state.result.artists}
				prob = {this.state.result.prob}
	
			/>;
		}

		return (
			<div>
				<div className="mt-5"/>
				{search_element}
				{error_info}
				{result_info}
			</div>
		)
	}

	render() {
		return (
			<body className="mainbody">
				<div className="h1 text-center">Is it Ska?</div>
				{this.state.loading ? this.loading() : this.content() }
				<div className="text-center">
					{this.version}
				</div>
			</body>
		);
	}
}
  
export default App;