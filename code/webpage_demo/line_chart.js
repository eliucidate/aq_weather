var city = "Choose your city";
var aq = "";
var path_prefix = "data/lines/";
var full_path = "data/lines/New Yorkco.csv";

$("#co").on("click", function(){
    city = $("#city").val();
    aq = "co";
    if (city != "Choose your city") {
      full_path = path_prefix + city + aq + ".csv";
    }
    draw(full_path)
});

$("#no2").on("click", function(){
    city = $("#city").val();
    aq = "no2";
    if (city != "Choose your city") {
      full_path = path_prefix + city + aq + ".csv";
    } 
    draw(full_path)
});

$("#pm25").on("click", function(){
    city = $("#city").val();
    aq = "pm25";
    if (city != "Choose your city") {
      full_path = path_prefix + city + aq + ".csv";
    }
    draw(full_path)
});

function draw(full_path) {
  $("#line_chart").html("");

  var margin = {top: 30, right: 20, bottom: 30, left: 50},
      width = 960 - margin.left - margin.right,
      height = 470 - margin.top - margin.bottom;

  var formatDate = d3.time.format("%m/%d/%Y").parse;

  var x = d3.time.scale()
      .range([0, width]);

  var y = d3.scale.linear()
      .range([height, 0]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom")
      .tickFormat(d3.time.format("%m/%d"));

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left");

  var line = d3.svg.line()
      .x(function(d) { return x(formatDate(d[0])); })
      .y(function(d) { return y(d[1]); });

  var svg = d3.select("#line_chart").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  function Comparator(a,b)
      { var ax = new Date(a[0]);
        var bx = new Date(b[0]);
        if (ax.getTime()<bx.getTime()) {return -1;}
        if (ax.getTime()>bx.getTime()) {return 1;}
        return 0;
      }
  var div = d3.select("#line_chart").append("div") 
      .attr("class", "tooltip")       
      .style("opacity", 0);

  d3.text(full_path, function(text){     
    var dict = [];
    data = d3.csv.parseRows(text);
    for (i = 0;i < data.length;i++){
      var date = new Date (((data[i][1]).split("on"))[1]);
      str =  (date.getMonth()+1)+"/"+date.getDate()+"/"+date.getFullYear();
      unit = data[0][4]
      if  (str in dict)
        { 
          dict[str].push(parseFloat(data[i][3]));
          
          }
      else{
        
        dict[str] = []
      dict[str].push(parseFloat(data[i][3]));}
      // console.log(dict[str]);
      }
      names = Object.keys(dict);
      dataf = []
      for (i = 0;i < names.length;i++){
        var sum = dict[names[i]].reduce(function(a,b) {return a+b;},0);
        dataf.push([names[i],sum/dict[names[i]].length]);

      }

    dataf = dataf.sort(Comparator);
    x.domain(d3.extent(dataf, function(d) { return formatDate(d[0]); }));
    y.domain(d3.extent(dataf, function(d) { return d[1]; }));

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
      .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text(unit);

    svg.append("path")
        .datum(dataf)
        .attr("class", "line")
        .attr("d", line);
    svg.selectAll("dot")
                  .data(dataf)
                .enter().append("circle")
                  .attr("r", 4)
                  .attr("cx", function(d) { return x(formatDate(d[0])); })
                  .attr("cy", function(d) { return y(d[1]); })
                  .on("mouseover", function(d) {    
              div.transition()    
                  .duration(200)    
                  .style("opacity", .9);    
              div.html(d[0] + "<br/>"  + d[1].toFixed(3))  
                  .style("left", (d3.event.pageX) + "px")   
                  .style("top", (d3.event.pageY - 28) + "px");  
              })          
          .on("mouseout", function(d) {   
              div.transition()    
                  .duration(500)    
                  .style("opacity", 0); 
          });;
  });
}
