import React from 'react';
import FixTrackButtons from './FixTrackButtons'
import '../style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class SearchResult extends React.Component {
	constructor(props) {
		super(props);
		
		this.track_id = this.props.track_id;
		this.track_link = this.props.track_link;
		this.title = this.props.title;
		this.artists = this.props.artists;
		this.prob = this.props.prob;

		this.state = {
			show_track_fix: false
		}
	}

	componentDidMount() {
	}

	buttonClicked = (text) => {
		if(text == undefined)
		{
			return	
		}

		this.search_callback(text);
	}

	
	getResultText(prob) {
		if(prob > 0.85) {
			return "Ska!";
		}

		if(prob > 0.75) {
			return "probably ska!";
		}

		if(prob > 0.5) {
			return "maybe is ska Probably maybe";
		}

		if(prob > 0.25) {
			return "not ska";
		}

		return "not ska you can't pick this one up.";
	}

	getArtistsText(artists) {
		let result = "";

		let i;
		for(i = 0; i < artists.length; i++) {
			result += artists[i];
			if(i < artists.length - 1) {
				result += i <= artists.length - 2 ? ", " : " and ";
			}
		}

		return result;
	}

	render() {
		const imgStyle = {
			"max-height" : "300px",
			"max-width": "300px",
			height: "auto",
			width : "auto"
		}

		var result = 
			<div className="text-center bg-white mr-5 ml-5">
				<div className="h4">
					The Track <a
						href={this.track_link} target="_blank" rel="noopener noreferrer" >{this.title}</a> by {this.getArtistsText(this.artists)}
				</div>
				<div className="h4">is</div>
				<div className="h3">{this.getResultText(this.prob)}</div>
				<FixTrackButtons
					className="mt-5"
					track_id = {this.track_id}
				/>
				{/* <img 
					src={data.album_image_url}
					style={imgStyle}
				/> */}
			</div>

		return result;
	}
}
  
export default SearchResult;