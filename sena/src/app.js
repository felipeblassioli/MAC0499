var React = require('react');
var List = require('material-ui/lib/lists/list');
var ListItem = require('material-ui/lib/lists/list-item');
var Checkbox = require('material-ui/lib/checkbox');

var ExperimentView = React.createClass({
	render: function(){
		var e;
		return (
			<div className={"experiment"}>
				<ExperimentDescription
					experiment={this.props.experiment} />

			</div>
		);
	}
});

var VisibilityOptions = React.createClass({
	render: function(){
		return (
			<div className={"mdl-card__supporting-text mdl-color-text--blue-grey-50"}>
				<h3>View options</h3>
				<ul>
					<li>
						<label htmlFor="chkbox1" className={"mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect"}>
							<input type="checkbox" id="chkbox1" className={"mdl-checkbox__input"} />
							<span className={"mdl-checkbox__label"}>Click per object</span>
						</label>
					</li>
				</ul>
			</div>
		);
	}
});

var ExperimentsList = React.createClass({
	getInitialState: function(){
		return {
			experiments: [
				{name: "Experiment A"},
				{name: "Experiment B"}
			]
		}
	},
	render: function(){
		var chk = <Checkbox name="xxx" />;
		return (
			<List>
				<ListItem primaryText="Experiment A" />
				<ListItem primaryText="Experiment A" />
				<ListItem primaryText="Experiment A" />
				<ListItem primaryText="Experiment A" />
				<ListItem primaryText="Experiment B" leftCheckbox={chk}/>
			</List>
		);
	}
});

var App = React.createClass({
	render: function(){
		return (
			<div className={"mdl-layout"}>
				<header className={"mdl-layout__header"}>
					<div className={"mdl-layout__header-row mdl-color--grey-100 mdl-color-text--grey-600 is-casting-shadow"}>
						<span className={"mdl-layout-title"}>
							Home
						</span>
					</div>
				</header>

				<main className={"mdl-layout__content mdl-color--grey-100"}>
					<div className={"mdl-grid"}>
						<div className={"mdl-cell mdl-cell--4-col"}>
							<ExperimentsList />			
						</div>
						<div className={"mdl-cell mdl-cell--4-col"}>
							<h1>hahhahah</h1>
						</div>
						<div className={"mdl-cell mdl-cell--4-col"}>
							<h1>hahhahah</h1>
						</div>
					</div>
					<div className={"mdl-grid"}>
						<div className={"mdl-cell mdl-cell--4-col"}>
							<VisibilityOptions />
						</div>
					</div>

				</main>

				<div className={"mdl-layout__obfuscator"}></div>
			</div>
		);
	}
});


module.exports = {
	init: function(elementId){
		React.render(<App />, document.getElementById(elementId));
	}
};
