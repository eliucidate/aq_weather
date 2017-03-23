pico.load('model');
pico.main = function() {
    var city;
    var temp;
    var wind_speed;
  
    var displayMessage = function(message){
       document.getElementById('message').innerHTML = message;
    }
    $("#submit").on("click", function(){
        city = $("#city").val();
        weather = $("#weather").val();
        temp = $("#inputBox1").val();
        wind_speed = $("#inputBox2").val();
        humi = $("#inputBox3").val();
        model.predict_1(city, weather, temp, wind_speed, humi, displayMessage);
    });
    $("#predict").on("click", function(){
        city = $("#city").val();
        model.predict_2(city, displayMessage);
    });       
}