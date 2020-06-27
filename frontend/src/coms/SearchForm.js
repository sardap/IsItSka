import React from 'react';
import '../style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class SearchForm extends React.Component {
	constructor(props) {
		super(props);

		this.search_callback = this.props.search_callback;

		this.state = {
			button_text: this.getSearchButtonText()
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

	buttonClicked = (text, artist_text) => {
		if(text == undefined)
		{
			return	
		}

		this.search_callback(text, artist_text);
	}

	render() {
		return (
			<div className="col text-center">
				<form>
					<div className="container">
						<div className="row justify-content-center">
							<label className="text-dark h2 p-3">Enter track name</label>
						</div>
						<div className="row mb-3">
							<input 
								className="form-control p-1 text-center ml-5 mr-5"
								onChange={(e) => this.setState({search_entry_text: e.target.value})}
								defaultValue={this.props.track_name}
								/>
						</div>
						<div className="row justify-content-center">
							<label className="text-dark h3 p-3">Enter artist name (optional)</label>
						</div>
						<div className="row mb-2">
							<input 
								className="form-control p-1 text-center ml-5 mr-5"
								onChange={(e) => { this.setState({artist_text: e.target.value})}}
								defaultValue={this.props.artist_name}
								/>
						</div>
					</div>
					<div class="form-group">
						<button 
							class={"border border-info btn btn-lg btn-light checkerboard-background-lg p-4 rounded"}
							type="button" 
							onClick={(e) => this.buttonClicked(this.state.search_entry_text, this.state.artist_text)}
							disabled={this.state.search_entry_text !== undefined && this.state.search_entry_text.length > 0 ? false : true}
							>
								<div className="text-black bg-white h3 rounded p-1">
									{this.state.button_text}
								</div>
						</button>
					</div>
				</form>
			</div>
		)
	}
}
  
export default SearchForm;