import React from 'react';
import '../style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class ResponseButton extends React.Component {
	constructor(props) {
		super(props);
		this.good = this.props.good;
		this.track_id = this.props.track_id;
		this.vote_callback = this.props.vote_callback;

		if(this.good) {
			this.color = "success";
			this.text = "Pick this up it's SKA!";
			this.ska = true;
		} else {
			this.color = "danger";
			this.text = "Drop this not ska garbage.";
			this.ska = false;
		}

	}

	componentDidMount() {
	}

	render() {
		return (
			<div className="text-center">
				<form>
					<div class="form-group">
						<button 
							class={"text-center border border-" + this.color + " btn btn-" + this.color}
							type="button" 
							onClick={(e) => this.vote_callback(this.track_id, this.ska)}
							>
								<div className="text-black">
									{this.text}
								</div>
						</button>
					</div>
				</form>
			</div>
		)
	}
}
  
export default ResponseButton;