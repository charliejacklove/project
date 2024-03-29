// Generated by Selenium IDE
import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.core.IsNot.not;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.edge.EdgeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.Alert;
import org.openqa.selenium.Keys;
import java.util.*;
import java.net.MalformedURLException;
import java.net.URL;
public class TransationsTest {
  private WebDriver driver;
  private Map<String, Object> vars;
  JavascriptExecutor js;
  @Before
  public void setUp() {
    driver = new EdgeDriver();
    js = (JavascriptExecutor) driver;
    vars = new HashMap<String, Object>();
  }
  @After
  public void tearDown() {
    driver.quit();
  }
  @Test
  public void transations() {
    driver.get("https://paraphraser.ai/");
    driver.manage().window().setSize(new Dimension(1552, 880));
    {
    List<WebElement> all2 = driver.findElements(By.id("paraphrase-input-box"));
    for(WebElement e:all2){
      if(e.getText().equals("paraphrase-input-box")){
        e.click();
        e.sendKeys("when do you finish your homework ?");
      }
    }}{
    List<WebElement> all3 = driver.findElements(By.id("paraphrase-submit"));
    for(WebElement e:all3){
      if(e.getText().equals("paraphrase-submit")){
        e.click();

      }
    }}{
    List<WebElement> all4 = driver.findElements(By.cssSelector("#paraphrase-input-box"));
    for(WebElement e:all4){
      if(e.getText().equals("#paraphrase-input-box")){
        e.click();

      }
    }}
    {
    List<WebElement> all7 = driver.findElements(By.cssSelector("#paraphrase-input-box"));
    for(WebElement e:all7){
      if(e.getText().equals("#paraphrase-input-box")){
        driver.findElement(By.cssSelector("#paraphrase-input-box")).sendKeys("when do you finish your homework ? it\'s very hard for me to change this courses.so i must have plenty of time to do it");


      }
    }}{
    List<WebElement> all1 = driver.findElements(By.cssSelector("#paraphrase-submit > span"));
    for(WebElement e:all1){
      if(e.getText().equals("#paraphrase-submit > span")){
        e.click();
      }
    }}
    {
    List<WebElement> all2 = driver.findElements(By.id("langs-dropdown"));
    for(WebElement e:all2){
      if(e.getText().equals("langs-dropdown")){
        e.click();
      }
    }}

    List<WebElement> all3 = driver.findElements(By.id("paraphrase-output-box"));
    for(WebElement e:all3){
      if(e.getText().equals("paraphrase-output-box")){
        e.click();
      }
    }


    js.executeScript("window.scrollTo(0,263.20001220703125)");

    List<WebElement> all4 = driver.findElements(By.id("paraphrase-input-box"));
    for(WebElement e:all3){
      if(e.getText().equals("paraphrase-input-box")){
        e.click();
      }
    }
    List<WebElement> all8 = driver.findElements(By.xpath("//textarea[@id=\'paraphrase-input-box\']"));
    for(WebElement e:all8){
      if(e.getText().equals("//textarea[@id=\'paraphrase-input-box\']")){
        driver.findElement(By.xpath("//textarea[@id=\'paraphrase-input-box\']")).sendKeys("when do you finish your homework ? it\'s very hard for me to change this courses.so i must have plenty of time to do it . i wonder how , i wonder why ,you told me like blue sky.");


      }
    }
    List<WebElement> all11 = driver.findElements(By.id("paraphrase-submit"));
    for(WebElement e:all3){
      if(e.getText().equals("paraphrase-submit")){
        e.click();
      }
    }

  }
}
