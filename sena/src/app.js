require('babel-polyfill');

injectTapEventPlugin = require("react-tap-event-plugin");
injectTapEventPlugin();

var React = require('react');
var List = require('material-ui/lib/lists/list');
var ListItem = require('material-ui/lib/lists/list-item');
var Checkbox = require('material-ui/lib/checkbox');
var Promise = require('bluebird');


var Reflux = require('reflux');
var _ = require('underscore');

var API = {
	fetchExperiments: function(){
		return new Promise(function(resolve, reject){
			var result = {data: [
				require('../data/result-bow-grabed'),
				require('../data/result-grabed')
			]};
			resolve(result);
		});
	}
};

/* Reflux stuff BEGIN */

var actions = {
	ExperimentsActions: Reflux.createActions({'refreshExperiments': {children: ['completed', 'failed'] }}),
	SelectedExperimentsActions: Reflux.createActions(['select', 'unselect'])
};

var stores = {
	ExperimentsStore: Reflux.createStore({
		listenables: [ actions.ExperimentsActions ],

		onRefreshExperimentsCompleted: function( experiments ){
			this.trigger( experiments );	
		}
	}),

	SelectedExperimentsStore: Reflux.createStore({
		listenables: [ actions.SelectedExperimentsActions ],

		init: function(){
			this._selectedExperiments = [];
		},

		onSelect: function( experiment ){
			this._selectedExperiments.push( experiment );
			this.trigger( this._selectedExperiments );
		},

		onUnselect: function( experiment ){
			this._selectedExperiments = _.without( this._selectedExperiments, experiment );
			this.trigger( this._selectedExperiments );
		}
	})
};

actions.ExperimentsActions.refreshExperiments.listen(function(){
	API.fetchExperiments().get('data').then(this.completed).catch(this.failed);	
});

/* Reflux stuff END */

_.mixin({
	sumObjects: function( listOfObjects ){
		return _.chain(listOfObjects).reduce( function( result, value, key ){
					_.each( value, function( v, k ){
						if( _.isNumber( v ) ){
							if( k in result )
								result[k] += v;
							else
								result[k] = v;
						}
					});
					return result;
				}, {} )
				.value();
	},

	meanObject: function( listOfObjects ){
		var count = _.size( listOfObjects );

		return _.chain(listOfObjects)
				.sumObjects( )
				.mapObject( function( v ){
					return v / count;
				}).value();
	},

	maxIndex: function( list ){
		return 	_.indexOf( list, _.max( list ) );
	},

	minIndex: function( list ){
		return 	_.indexOf( list, _.min( list ) );
	},

	isFloat: function( n ){
		return n === Number(n) && n % 1 !== 0;
	}

});

var Card = require('material-ui/lib/card/card');
var CardHeader = require('material-ui/lib/card/card-header');
var CardTitle = require('material-ui/lib/card/card-title');

var Table = require('material-ui/lib/table/table');
var TableHeader = require('material-ui/lib/table/table-header');
var TableHeaderColumn = require('material-ui/lib/table/table-header-column');
var TableBody = require('material-ui/lib/table/table-body');
var TableRow = require('material-ui/lib/table/table-row');
var TableRowColumn = require('material-ui/lib/table/table-row-column');

var ArrayTable = React.createClass({

	getDefaultProps: function(){
		return {
			columnLabels: [],
			title: ""
		};
	},

	_renderValue: function( col ){
		if( _.isObject( col ) ){
			if( col.isLabel )
				return <strong>{ col.value }</strong>;
			if( this.props.renderValue )
				return this.props.renderValue( col );
		}
		return col;
	},

	_renderColumn: function( rowIndex, col, colIndex ){
		var _valueToString = this.props.valueToString || function( v ){ return v; };
		var _renderValue = this._renderValue; 
		return <TableRowColumn key={'col-'+rowIndex+'-'+ colIndex}>{ _renderValue( _valueToString( col ) ) }</TableRowColumn>;
	},

	_renderRow: function( values, rowIndex ){
		return (
			<TableRow key={'row-'+rowIndex}>
				{ _.map( values, this._renderColumn.bind( this, rowIndex )) }
			</TableRow>
		);
	},

	render: function(){
		var _data = this.props.data;
		var _toLabel = function( v ){ return { value: v, isLabel: true } };

		if( _.isArray( _data ) ){
			var _rowLabels = this.props.rowLabels || [];
			_rowLabels = _.map( _rowLabels, _toLabel );
			if( _.size( _rowLabels ) > 0 )
				_data = _.map( _.zip( _rowLabels, _data), function( t ){ return [ t[0] ].concat( t[1] ); } );
		} else if( _.isObject( _data ) ) {
			_data = _.map( _data, function( v, k ){
				return _.flatten([ _toLabel( k ), v ]);
			});
			if( this.props.cols ){
				var _cols = _.size(_data) / this.props.cols;
				var chunks = _.chain(_data).groupBy(function(e,index){ return Math.floor(index/_cols);}).toArray().value();
				_data = _.reduce( chunks, function( result, array ){
					result = _.map( _.zip( result, array ), function( t ){
						return t[0].concat(t[1]);
					});
					return result;
				});
			}
		}

		return (
			<Table selectable={false}>
				{ this.props.title? 
					<TableHeader displaySelectAll={false}>
						<TableRow>
							<TableHeaderColumn colSpan={_.size(_.first(_data))} style={{textAlign: 'center'}}>
								<strong>{ this.props.title }</strong>
							</TableHeaderColumn>
						</TableRow>
					</TableHeader>
				: null
				}
				<TableBody displayRowCheckbox={false}>
					{ this.props.columnLabels.length > 0?
						<TableRow key={'label-row'}>
							{
								_.map( this.props.columnLabels, function( label, index ){
									return (
										<TableRowColumn key={'colHeader-'+index} style={{textAlign: 'center'}}>
											<strong>{ label }</strong>
										</TableRowColumn>
									);
								}) 
							}
						</TableRow>
					: null }
					{ _.map( _data, this._renderRow ) }
				</TableBody>
			</Table>
		);
	}
});

var StatisticsTable = React.createClass({

	getDefaultProps: function(){
		return {
			title: "Statistics",
			cols: 1
		}
	},

	_valueToString: function( v ){
		if( _.isFloat( v ) )
			return (v * 100.0).toFixed(1) + ' %';
		else
			return v;
	},

	render: function(){
		var _valueToString = this.props.valueToString || this._valueToString;
		return (
			<ArrayTable
				{...this.props}
				title={ this.props.title }
				data={ this.props.data }
				cols={ this.props.cols }
				valueToString={ _valueToString }
				/>
		);
	}
});

var MaxMinTable = React.createClass({

	_valueToString: function( v ){
		if( _.isObject( v ) )
			v = v.value;
		if( _.isFloat( v ) )
			return (v * 100.0).toFixed(1) + ' %';
		return v;
	},

	_renderValue: function( o ){
		if( o.isMax )
			return <span style={{color: 'green'}}>{ this._valueToString( o.value ) }</span>;
		else if( o.isMin )
			return <span style={{color: 'red'}}>{ this._valueToString( o.value ) }</span>;
		else
			return o.value
	},

	render: function(){
		//var _renderValue = this.props.renderValue || this._renderValue;
		var _data = _.map( this.props.data, function( values, key ){ return [ key ].concat(values); } );	
		var maxIndex = [];
		var minIndex = [];
		var _newData = [];
		_.each( _.range( 1, _.size( _data[0] ) ), function( index ){
			var list = _.map( _data, function( arr ){ return arr[ index ]; } );
			maxIndex = _.maxIndex( list );
			minIndex = _.minIndex( list );

			list[ maxIndex ] = { value: list[ maxIndex ], isMax: true };
			list[ minIndex ] = { value: list[ minIndex ], isMin: true };
			_newData.push( list );
		});

		_newData = _.zip.apply( null, _newData );
		_data = _.object( _.zip( _.map( _data, _.first ), _newData ) );
		return <StatisticsTable {...this.props} data={_data} renderValue={this._renderValue} />; 
	}
});

var Tabs = require('material-ui/lib/tabs/tabs');
var Tab = require('material-ui/lib/tabs/tab');

var GridList = require('material-ui/lib/grid-list/grid-list');
var GridTile = require('material-ui/lib/grid-list/grid-tile');

var Experiment = {
	getOverviewData: function( sample ){
		_.each( sample.data, function( d ){
			d.predictedCorrectly = ( d.predictedLabel == sample.label );
		});
		var _incorrectlyPredicted = _.filter( sample.data, function( d) { return !d.predictedCorrectly } );
		return {
			Error: _.size( _incorrectlyPredicted ) / _.size( sample.data ),
			Precision: sample['statistics']['PPV'],
			Recall: sample['statistics']['TPR']
		};
	},

};

var ExperimentView = React.createClass({

	_renderSummary: function( samplesData ){
		var confusionMatrixData = this.props.experiment.confusion_matrix;
		var categories = _.map(this.props.experiment.training_dataset.categories, function( v ){
			return [ parseInt( v[0] ) - 1, v[1] ];
		});

		var stats2 = _.sumObjects( _.values( samplesData ) );
		stats2 = [
			[ stats2[ 'TP' ] + ' True positives', stats2[ 'FP' ] + ' False negatives' ],
			[ stats2[ 'FN' ] + ' False negatives', stats2[ 'TN' ] + ' True negatives' ]
		];

		var stats = _.meanObject( _( samplesData ).map(_.property('statistics')) );

		return ( 
			<Card initiallyExpanded={true}>
				<CardTitle title="Summary" showExpandableButton={true} />
				<div className="mdl-grid" expandable={true}>
					<div className="mdl-cell mdl-cell--6-col">
						<div className="mdl-grid">
							<ArrayTable title="Confusion Matrix" rowLabels={ _.object(categories) } data={confusionMatrixData} />
							<ArrayTable title="Table of Confusion" data={ stats2 } />
						</div>
					</div>
					<div className="mdl-cell mdl-cell--6-col">
						<StatisticsTable data={stats} cols={2} />
					</div>
				</div>
			</Card>
		);
	},

	_renderClassOverview: function( samplesData ){
		var _columnHeaders = _.keys( samplesData[0].statistics );
		var _samplesData = _.map( samplesData, function( sample ){
			var _data = _.map( _columnHeaders, function( h ){
				return sample.statistics[ h ];
			});

			return [ sample.label, _data ]
		});
		_samplesData = _.object( _samplesData );
		_columnHeaders = [""].concat( _columnHeaders );

		var _overviewLabels = [""].concat(_.keys( Experiment.getOverviewData( samplesData[0] ) ));
		var _overviewData =  _.map( samplesData, function( sample ){
			return [ sample.label, _.values( Experiment.getOverviewData( sample ) ) ];
		});
		_overviewData = _.object( _overviewData );

		var _confusionLabels = ['TP', 'FP', 'FN', 'TN'];
		var _confusionData = _.map( samplesData, function( sample ){
			return [ sample.label, _.map( _confusionLabels, function( h ){ return parseInt(sample[ h ]); }) ];	
		});
		_confusionData = _.object( _confusionData );
		_confusionLabels = [""].concat(_confusionLabels);
		return (
			<Card initiallyExpanded={true}>
				<CardTitle title="Class Overview" showExpandableButton={true} />
				<div className="mdl-grid" expandable={true}>
					<div className="mdl-cell mdl-cell--6-col">
						<MaxMinTable title="Overview" columnLabels={_overviewLabels} data={_overviewData} />
					</div>
					<div className="mdl-cell mdl-cell--6-col">
						<MaxMinTable title="Table of Confusion" columnLabels={_confusionLabels} data={_confusionData} />
					</div>
				</div>
				<MaxMinTable expandable={true} title="Statistics" columnLabels={_columnHeaders} data={ _samplesData } />
			</Card>
		);
	},

	_renderResults: function( _samplesData, _labels ){

		var _renderTab = function( sample, index ){
			_.each( sample.data, function( d ){
				d.predictedCorrectly = ( d.predictedLabel == sample.label );
			});

			var _renderGridTile = function( d, index ){
				var _style;
				if( d.predictedCorrectly )
					_style = { backgroundColor: "green" };
				else
					_style = {  backgroundColor: "red" };

				return <GridTile
							key={'tile-'+index}
							title={ d.predictedLabel }
							titlePosition="top"
							style={_style}>

							<img src={ d.imageUrl } />

						</GridTile>;
			};

			var _correctlyPredicted = _.filter( sample.data, _.property('predictedCorrectly') );
			var _incorrectlyPredicted = _.filter( sample.data, function( d) { return !d.predictedCorrectly } );

			var overviewData = Experiment.getOverviewData( sample );

			return (
				<Tab key={'tab'+index} label={ sample.label }>

					<div className="mdl-grid">
					
						<div className="mdl-cell mdl-cell--6-col">
							<StatisticsTable title="Overview" data={overviewData} />
							<ArrayTable title="Table of Confusion" data={ sample.confusionTable } /> 	
						</div>

						<div className="mdl-cell mdl-cell--6-col">
							<StatisticsTable data={ sample.statistics } cols={2} />
						</div>

						<div className="mdl-cell mdl-cell--6-col">
							<h3>Correct ( { _.size( _correctlyPredicted ) } )</h3>
							<GridList
								cols={4}
								padding={1} >
								
								{ _.map( _correctlyPredicted, _renderGridTile ) }

							</GridList>
						</div>
						
						
						<div className="mdl-cell mdl-cell--6-col">
							<h3>Incorrect ( { _.size( _incorrectlyPredicted ) } )</h3>
							<GridList
								cols={4}
								padding={1} >
								
								{ _.map( _incorrectlyPredicted, _renderGridTile ) }

							</GridList>
						</div>
					</div>

				</Tab>
			);
		};

		return (
			<Card initiallyExpanded={true}>
				<CardTitle title="Details" showExpandableButton={true} />
				<Tabs expandable={true}>
					{ _.map( _samplesData, _renderTab ) }
				</Tabs>
			</Card>
		);

	},

	render: function(){
		var _labels = _.map(this.props.experiment.training_dataset.categories, function( v ){
			return [ parseInt( v[0] ) , v[1] ];
		});
		_labels = _.object( _labels );

		/*var _samplesData = _.map( this.props.experiment.samples, function( v, k ){
			v.label = _labels[ parseInt(k) ] || k;
			return v;
		});*/
		var _samplesData = _.map( this.props.experiment.samples, function( v, k ){
			// TODO: Now this idea of using 'tuples' instead of object was stupid to say the least
			var _data = _.map( v.data, function( ans ){
				//TODO: fucking gambiarra
				if( _.isArray( ans ) )
					return {
						imageUrl: ans[0],
						predictedLabel: _labels[ parseInt(ans[1]) ]
					};
				else
					return ans;
			});

			v.label = _labels[ parseInt(k) ] || k;
			v.data = _data;
			v.confusionTable = [
				[ v[ 'TP' ] + ' True positives', v[ 'FP' ] + ' False positives' ],
				[ v[ 'FN' ] + ' False negatives', v[ 'TN' ] + ' True negatives' ]
			];
			return v;
		});

		return (
			<Card initiallyExpanded={true}>
				<CardHeader title={ this.props.experiment.algorithm_name }> </CardHeader>
				{ this._renderSummary( _samplesData ) }
				{ this._renderClassOverview( _samplesData ) }
				{ this._renderResults( _samplesData, _labels ) }
			</Card>
		);
	}
});

var LeftNav = require('material-ui/lib/left-nav');
var MenuItem = require('material-ui/lib/menus/menu-item');
var MenuDivider = require('material-ui/lib/menus/menu-divider');
var FontIcon = require('material-ui/lib/font-icon');

var ArrowDropRight = require('material-ui/lib/svg-icons/navigation-arrow-drop-right');
var LineChart = require('react-d3').LineChart;

var App = React.createClass({

	mixins: [ Reflux.connect( stores.ExperimentsStore, 'experiments' ) ],

	getInitialState: function(){
		return {
			experiments: [],
			selectedIndex: 0,
			mode: 'details',
			isVisible: {
				'ACC': true,
				'PPV': true,
				'TPR': true,
				'F1': false,
				'FNR': false,
				'TNR': false,
				'MCC': false,
				'FDR': false,
				'FPR': false,
				'NPV': false,
				'TPR': false
			}
		};
	},

	_renderExperiment: function( experiment, index ){
		if( index > 0 )
			return null;
		return (
			<div className={"mdl-cell mdl-cell--8-col"}>
				<ExperimentView experiment={ experiment } />			
			</div>
		);
	},

	_changeMode: function( mode ){
		this.setState({ mode: mode });
	},

	_renderLeftNav: function(){
		var _menuHeader = <h4>MAC0499 - Demo</h4>;
		var that = this;
		
		var _onExperimentClick = function( index, evt ){
			that.setState({ selectedIndex: index });
		};

		return (
			<LeftNav 
				ref="leftNav" 
				className="mdl-layout__drawer"
				header={_menuHeader} >
				
				<h6>Tools</h6>

				<MenuItem index={0} leftIcon={<FontIcon className="material-icons">visibility</FontIcon>} onClick={this._changeMode.bind(this, 'details')}>Details </MenuItem>
				<MenuItem index={1} leftIcon={<FontIcon className="material-icons">assessment</FontIcon>} onClick={this._changeMode.bind(this, 'analysis')}>Analysis </MenuItem>

				<h6>Experiments</h6>

				{
					_.map( this.state.experiments, function( e, index ){
						return <MenuItem key={'menuItem'+index} index={2 + index} leftIcon={<ArrowDropRight />} onClick={_onExperimentClick.bind( this, index )}>{ e.algorithm_name }</MenuItem>
					})

				}

			</LeftNav>
		);

	},

	_renderMain: function(){
		var _content;
		if( this.state.mode === 'details' ){
			var _currentExperiment = _.size( this.state.experiments ) > 0? this.state.experiments[ this.state.selectedIndex ] : null;
			console.log('_currentExperiment is', _currentExperiment );
			if( ! _currentExperiment )
				return null;
			_content = this._renderExperiment( _currentExperiment );
		} else {

			var mainData = _.map( this.state.experiments, function( e ){
				return {
					name: e.algorithm_name,
					//TODO: Write ExperimentWrapper and remove this duplicate code
					statistics: _.meanObject( _( e.samples ).map( _.property('statistics') ) ) 
				}
			});

			var _isVisible = this.state.isVisible;
			var options = _.keys( _isVisible );
			var _visibleOptions = _.filter( options, function( k ){ return _isVisible[ k ]; } ); 
			console.log('_visibleOptions', _visibleOptions);
			var _lineData = _.map( _visibleOptions, function( opt ){
				return {
					name: opt,
					values: _.map( mainData, function( e, index ){ return { x: index * 0.2, y: e.statistics[opt] }; } )
				};
			});

			var _handleOnCheck = function( option, evt, isChecked ){
				this.state.isVisible[ option ] = isChecked;
				this.forceUpdate();
			}.bind(this);

			_content = (
				<div className={"mdl-cell mdl-cell--8-col"}>
					<Card initiallyExpanded={true}> 
						<div className="mdl-grid">
							<div className="mdl-cell mdl-cell--2-col">
								<List subheader="Visible Options">
									{ 
										_.map( options, function( o, index ){
											return <ListItem 
														leftCheckbox={<Checkbox defaultChecked={_isVisible[o]} onCheck={_handleOnCheck.bind(this, o)}/>}
														key={'opt-'+index} 
														primaryText={o} />;
										})
									}
								</List>
							</div>

							<div className="mdl-cell mdl-cell--8-col">
								<LineChart
									title="Statistics"
									legend={true}
									data={_lineData}
									circleRadius={6}
									width={900}
									height={800}
									gridHorizontal={true} />
							</div>
						</div>
					</Card>
				</div>
			);
		}
		return (
			<main className={"mdl-layout__content mdl-color--grey-100"}>
				<div className={"mdl-grid"}>
					{ _content }
				</div>
			</main>
		)
	},

	render: function(){
		var _menuItems = [
			{ text: "Components", route: "bla" },
			{ text: "Components", route: "bla" }
		];
		console.log('experiments', this.state.experiments);
		console.log('App.render', this.state);
		console.log('current mode is', this.state.mode);
		var _currentTitle = this.state.mode === 'details'? 'Experiment Details' : 'Experiment Analysis';
		return (
			<div className="mdl-layout__container">
			<div className={"mdl-layout mdl-layout--fixed-drawer mdl-layout--fixed-header has-drawer is-upgraded"}>
				<header className={"mdl-layout__header"}>
					<div className={"mdl-layout__header-row mdl-color--grey-100 mdl-color-text--grey-600 is-casting-shadow"}>
						<span className={"mdl-layout-title"}>
							{ _currentTitle }
						</span>
					</div>
				</header>

				{ this._renderLeftNav() }
				{ this._renderMain() }
			</div>
			</div>
		);
	}
});

actions.ExperimentsActions.refreshExperiments();
module.exports = {
	init: function(elementId){
		React.render(<App />, document.getElementById(elementId));
	}
};
