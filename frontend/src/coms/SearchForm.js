import React from 'react';
import '../style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class SearchForm extends React.Component {
	constructor(props) {
		super(props);

		this.search_callback = this.props.search_callback;

		this.state = {
			button_text: this.getSearchButtonText(),
			show_track_search: this.props.track_id === undefined
		};
	}

	getSearchButtonText() {
		const messages = [
			"Pick it up!",
			"Hey hey HEY!",
			"Everybody let's go!",
			"Let's Skank!",
			"Pick it the FUCK up!",
			"Ska came before reggae!",
			"Dee dee don't like ska!",
			"Is it Ska? Maybe?",
			"Ska defines who I am!"
		];
		return messages[Math.floor(Math.random() * messages.length)];
	}

	componentDidMount() {
		if(this.props.track_name !== undefined) {
			this.setState({
				search_entry_text: this.props.track_name
			});
		}

		if(this.props.artist_name !== undefined) {
			this.setState({
				artist_name: this.props.artist_name
			});
		}
	}

	buttonClicked = (text, artist_name, track_id) => {
		if(text == undefined && track_id == undefined)
		{
			return	
		}

		this.search_callback(text, artist_name, track_id);
	}

	renderSearchByName() {
		return (
			<div className="container">
				<div className="row justify-content-center">
					<label className="text-dark h2 p-3">Enter track name</label>
				</div>
				<div className="row mb-3">
					<input 
						className="form-control p-1 text-center ml-5 mr-5"
						onChange={(e) => this.setState({search_entry_text: e.target.value})}
						defaultValue={this.props.track_name}
						value={this.state.search_entry_text}
						/>
				</div>
				<div className="row justify-content-center">
					<label className="text-dark h3 p-3">Enter artist name (optional)</label>
				</div>
				<div className="row mb-2">
					<input 
						className="form-control p-1 text-center ml-5 mr-5"
						onChange={(e) => { this.setState({artist_name: e.target.value})}}
						defaultValue={this.props.artist_name}
						value={this.state.artist_name}
						/>
				</div>
			</div>
		);
	}

	renderSearchByID() {
		return (
			<div className="container">
				<div className="row justify-content-center">
					<label className="text-dark h2 p-3">Enter Spotify track URL</label>
				</div>
				<div className="row mb-5">
					<input 
						className="form-control p-1 text-center ml-5 mr-5"
						onChange={(e) => this.setState({track_id: e.target.value})}
						defaultValue={this.props.track_id}
						value={this.state.track_id}
						/>
				</div>
			</div>
		)
	}

	toggleFields = async () => {
		let next_state = !this.state.show_track_search;

		if(!this.state.show_track_search) {
			await this.setState({
				track_id: null,
				show_track_search: next_state
			});
		} else {
			await this.setState({
				search_entry_text: null,
				artist_name: null,
				show_track_search: next_state
			});			
		}

		console.log("toggled: " + JSON.stringify(this.state));
	}

	render() {
		return (
			<div className="col text-center">
				<form>
					<div class="custom-control custom-switch">
						<input
							type="checkbox"
							class="custom-control-input"
							id="customSwitch1"
							checked={!this.state.show_track_search}
							onChange={(e) => { this.toggleFields(); }}
						/>
						<label class="custom-control-label" for="customSwitch1">Use Spotify link</label>
					</div>
					{this.state.show_track_search ? this.renderSearchByName() : this.renderSearchByID() }
					<div class="form-group mt-3">
						<button 
							class={"border border-info btn btn-lg btn-light checkerboard-background-lg p-4 rounded"}
							type="button" 
							onClick={(e) => this.buttonClicked(this.state.search_entry_text, this.state.artist_name, this.state.track_id)}
							disabled={
								!(
									(
										this.state.search_entry_text !== undefined &&
										this.state.search_entry_text !== null &&
										this.state.search_entry_text.length > 0
									)
									|| 
									(
										this.state.track_id !== undefined &&
										this.state.track_id !== null &&
										this.state.track_id.length > 0
									)
								) ? true : false
							}
							>
								<div className="text-black bg-white rounded p-1">
									<div className="h3">
										Search
									</div>
									<div className="h4">{this.state.button_text}</div>
								</div>
						</button>
					</div>
				</form>
			</div>
		)
	}
}
  
export default SearchForm;