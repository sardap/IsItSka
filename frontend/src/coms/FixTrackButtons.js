import React from 'react';
import ResponseButton from './ResponseButton'
import '../style/App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

class FixTrackButtons extends React.Component {
	constructor(props) {
		super(props);

		this.track_id = this.props.track_id;
		this.vote_callback = this.props.vote_callback;

		if(this.props.show !== undefined) {
			this.state = {
				show: this.props.show
			};
		} else {
			this.state = {
				show: false
			};
		}
		
	}

	componentDidMount() {
	}

	voteCallback = async (track_id, ska) => {
		let url = window.location.origin + "/api/correction";


		const formData = new FormData();

		formData.append('track_id', track_id);
		if(ska) {
			formData.append('ska', true);
		}

		console.log("Making Request: " + url);
		try {
			let response = await fetch(url, {
				method: "POST",
				body: formData
			});
			if(response.status == 200) {
				this.setState({
					show: false,
					show_message: "Thanks for your feedback!"
				});
			} else if(response.status == 404) {
				console.log("error adding correction");
			} else {
				console.log("Error: " + JSON.stringify(response));
			}
		} catch(e) {
			console.log(e);
		}

		
	}

	buttonClicked = () => {
		this.setState({
			show: !this.state.show,
			show_message: undefined
		});
	}

	showResponseButtons() {
		const result = 
			<>
				<div className="col">
					<ResponseButton 
						good={true}
						track_id={this.track_id}
						vote_callback={this.voteCallback}
					/>
				</div>
				<div className="col">
					<ResponseButton
						good={false}
						track_id={this.track_id}
						vote_callback={this.voteCallback}
					/>
				</div>
			</>;

		return result;
	}

	showShowButton() {
		const result = 
			<>
				<button 
					class={"border border-info btn btn-info"}
					type="button" 
					onClick={(e) => this.buttonClicked()}
					>
						<div className="text-black">
							This doesn't seem right
						</div>
				</button>
			</>;

		return result;
	}

	render() {
		let message = <></>;

		if(this.state.show_message) {
			message = 
				<>
					<span class="badge badge-pill badge-success">{this.state.show_message}</span>
				</>
		}

		const result =
			<div>
				<div className={"container " + this.props.className}>
					<div className="row justify-content-center">
						{this.state.show ? this.showResponseButtons() : this.showShowButton() }
					</div>
				</div>
				<div className="mt-3" />
				{message}
			</div>

		return result;
	}
}
  
export default FixTrackButtons;